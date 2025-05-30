#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import paramiko
import fire
from colorama import Fore, Style, init
import json
import re
import fnmatch
from sshconf import read_ssh_config

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Initialize colorama
init()

IGNORE_HOSTS = [
    "jumphost",  # Example of a host to ignore
    "dev-dsk-*.amazon.com",
    "*.corp.amazon.com",
    "git.amazon.com",
    "github.audible.com",
    "hyper-ai-*",
]


def ignore_host(hostname):
    if hostname in IGNORE_HOSTS:
        return True
    for ignore in IGNORE_HOSTS:
        if fnmatch.fnmatch(hostname, ignore):
            return True
    return False


def get_gpu_status(client):
    """Execute nvidia-smi command and return results"""
    try:
        stdin, stdout, stderr = client.exec_command(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
            timeout=10,
        )
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        if error:
            print(f"Error executing nvidia-smi: {error}")
            return f"Error: {error}"
        return output
    except Exception as e:
        print(f"Failed to execute command: {str(e)}")
        return f"Command execution failed: {str(e)}"


def parse_nvidia_smi_output(output, hostname):
    """Parse nvidia-smi output, extract GPU usage information"""
    try:
        gpu_info = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            # Parse CSV line
            parts = line.split(", ")
            if len(parts) < 5:
                continue

            gpu_id = int(parts[0])
            gpu_name = parts[1]
            used_memory = int(parts[2])
            total_memory = int(parts[3])
            util = int(parts[4])

            # Check if GPU is free (low utilization)
            is_free = util < 5

            gpu_info.append(
                {
                    "hostname": hostname,
                    "gpu_id": gpu_id,
                    "name": gpu_name,
                    "memory_used": used_memory,
                    "memory_total": total_memory,
                    "utilization": util,
                    "is_free": is_free,
                    "processes": 0,  # No longer tracking processes
                }
            )

        return gpu_info
    except Exception as e:
        print(f"Failed to parse nvidia-smi output: {str(e)}")
        print(f"Original output: {output}")
        return []


def load_ssh_config():
    """Load SSH config file using sshconf"""
    config_file = os.path.expanduser("~/.ssh/config")

    if not os.path.exists(config_file):
        return None

    try:
        config = read_ssh_config(config_file)

        wildcard_config_dict = {}
        for host in config.hosts():
            if "*" in host and not ignore_host(host):
                wildcard_config_dict[host] = config.host(host)

        # If wildcard config exists, merge it with specific host configs
        if wildcard_config_dict:
            for host in config.hosts():
                if "*" in host or ignore_host(host):
                    continue
                for wildcard_host, wildcard_config in wildcard_config_dict.items():
                    if fnmatch.fnmatch(host, wildcard_host):

                        # Get the specific host config
                        specific_config = config.host(host)

                        # Create merged config prioritizing specific over wildcard
                        merged_config = dict(wildcard_config)
                        merged_config.update(specific_config)

                        # Update host config in one call
                        try:
                            config.set(host, **merged_config)
                        except ValueError as e:
                            print(f"Failed to update config for host {host}: {e}")

        return config
    except Exception as e:
        print(f"Failed to load SSH config: {str(e)}")
        return None


def check_gpu_server(hostname, ssh_config, timeout=10):
    """Connect to server using SSH config and check GPU status"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Get host configuration
        host_config = ssh_config.host(hostname)

        # Extract settings
        real_hostname = host_config.get("hostname")
        username = host_config.get("user")
        port = int(host_config.get("port", 22))

        # Handle ConnectTimeOut (case-sensitive)
        connect_timeout = timeout
        if "ConnectTimeOut" in host_config:
            connect_timeout = int(host_config["ConnectTimeOut"])

        # Handle ForwardAgent (case-sensitive)
        forward_agent = host_config.get("ForwardAgent", "").lower() == "yes"

        # Connect to server
        client.connect(
            hostname=real_hostname,
            port=port,
            username=username,
            timeout=connect_timeout,
            allow_agent=forward_agent,
            banner_timeout=connect_timeout,
            auth_timeout=connect_timeout,
            channel_timeout=connect_timeout,
            disabled_algorithms=(
                {"pubkeys": [host_config.get("PubkeyAcceptedKeyTypes")]}
                if host_config.get("PubkeyAcceptedKeyTypes")
                else None
            ),
        )

        # Get GPU status with nvidia-smi
        output = get_gpu_status(client)

        if "Error" in output or not output:
            print(f"Unable to get GPU info on server {hostname}")
            return [
                {
                    "hostname": hostname,
                    "gpu_id": -1,
                    "name": "N/A",
                    "memory_used": -1,
                    "memory_total": -1,
                    "utilization": -1,
                    "is_free": False,
                    "processes": -1,
                    "error": output,
                }
            ]
        else:
            # Parse nvidia-smi output
            return parse_nvidia_smi_output(output, hostname)

    except Exception as e:
        print(f"Failed to connect to server {hostname}: {str(e)}")
        return [
            {
                "hostname": hostname,
                "gpu_id": -1,
                "name": "N/A",
                "memory_used": -1,
                "memory_total": -1,
                "utilization": -1,
                "is_free": False,
                "processes": -1,
                "error": str(e),
            }
        ]
    finally:
        client.close()


def print_gpu_status(all_gpu_info, only_free=False, brief=False):
    """Print GPU status information, highlight free GPUs"""
    if not all_gpu_info:
        print("No GPU information found")
        return

    # Group by server
    servers = {}
    for gpu in all_gpu_info:
        hostname = gpu["hostname"]
        if hostname not in servers:
            servers[hostname] = []
        servers[hostname].append(gpu)

    # Print GPU info for each server
    max_hostname_len = max(len(host) for host in servers.keys())
    for hostname, gpus in servers.items():
        if brief:
            free_gpus = 0
            busy_gpus = 0
            full_gpus = 0

            error_flag = False
            for gpu in gpus:
                if gpu.get("error"):
                    error_flag = True

                mem_ratio = gpu["memory_used"] / gpu["memory_total"]

                if mem_ratio > 0.5:
                    full_gpus += 1
                elif mem_ratio > 0.01:
                    busy_gpus += 1
                else:
                    free_gpus += 1

            if only_free and free_gpus == 0:
                continue

            tag = "⭐" if free_gpus == len(gpus) else "  "
            _hostname_str = f"{tag} {hostname+':':<{max_hostname_len}}"
            print(f"{Fore.CYAN}{_hostname_str}{Style.RESET_ALL}", end=" ")

            if error_flag:
                print(f"{Fore.RED}Error retrieving GPU info{Style.RESET_ALL}")
            else:
                print(
                    f"({Fore.GREEN if free_gpus > 0 else Style.DIM}Free: {free_gpus}{Style.RESET_ALL}, "
                    f"{Fore.YELLOW if busy_gpus > 0 else Style.DIM}Busy: {busy_gpus}{Style.RESET_ALL}, "
                    f"{Fore.RED if full_gpus > 0 else Style.DIM}Full: {full_gpus}{Style.RESET_ALL})"
                )
        else:
            print(f"\n{Fore.CYAN}======== Server: {hostname} ========{Style.RESET_ALL}")
            print(
                f"{'GPU ID':6} {'GPU Name':20} {'Memory Usage':25} {'Mem %':8} {'Util %':8} {'Status':6}"
            )

            for gpu in gpus:
                # Calculate memory usage ratio
                mem_ratio = gpu["memory_used"] / gpu["memory_total"]

                # Set color based on memory usage
                if mem_ratio > 0.5:
                    color = Fore.RED
                    status = "In Use"
                elif mem_ratio > 0.01:
                    color = Fore.YELLOW
                    status = "In Use"
                else:
                    color = Fore.GREEN
                    status = "Free"

                memory_str = f"{gpu['memory_used']}MiB / {gpu['memory_total']}MiB"
                mem_ratio_str = f"{mem_ratio:.1%}"
                util_str = f"{gpu['utilization']}%"

                if only_free and status != "Free":
                    continue

                print(
                    f"{color}{gpu['gpu_id']:6} {gpu['name'][:20]:20} {memory_str:25} {mem_ratio_str:8} {util_str:8} {status:6}{Style.RESET_ALL}"
                )


def get_servers(pattern, ssh_config):
    servers = []
    for host in ssh_config.hosts():
        if ignore_host(host):
            continue
        if "*" in host:
            continue
        if fnmatch.fnmatch(host, pattern):
            servers.append(host)
    if not servers:
        return None
    return servers


def main(
    servers="*", timeout=10, only_free=False, refresh=False, brief=False, interval=5
):
    """Check GPU usage on remote servers

    Args:
        servers: List of server aliases to check, separated by spaces
        timeout: SSH connection timeout in seconds
    """
    # Load SSH config
    ssh_config = load_ssh_config()
    if not ssh_config:
        print("Failed to load SSH config file")
        sys.exit(1)

    # Get list of servers to check
    servers = get_servers(servers, ssh_config)

    if servers is None:
        print("Please provide a valid server pattern.")
        sys.exit(1)
    elif not isinstance(servers, list):
        print("Invalid server list, must be a list of server aliases.")
        sys.exit(1)

    async def monitor_gpus():
        while True:
            all_gpu_info = []
            # Use ThreadPoolExecutor for parallel SSH connections
            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                # Create tasks for each server
                tasks = []
                for hostname in servers:
                    task = asyncio.get_event_loop().run_in_executor(
                        executor, check_gpu_server, hostname, ssh_config, timeout
                    )
                    tasks.append(task)

                # Gather all results
                results = await asyncio.gather(*tasks)
                for gpu_info in results:
                    all_gpu_info.extend(gpu_info)

            # Print GPU status
            os.system("clear")  # Clear screen on Mac/Unix systems
            print(f"Refreshing GPU status... ({time.strftime('%Y-%m-%d %H:%M:%S')})")
            print_gpu_status(all_gpu_info, only_free, brief)

            if not refresh:
                break

            await asyncio.sleep(interval)

    try:
        # Run the monitoring loop
        asyncio.run(monitor_gpus())
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        sys.exit(0)


if __name__ == "__main__":
    fire.Fire(main)
