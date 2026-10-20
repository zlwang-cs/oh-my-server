# oh-my-server
How to quickly set up a new server

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/zlwang-cs/oh-my-server/main/new_server_setup.sh | bash
# or, non-interactive: ... | bash -s -- --all
```

`new_server_setup.sh` asks which apps to install, then installs each one together with its config:

| App   | What you get |
|-------|--------------|
| zsh   | zsh + oh-my-zsh, plugins `git zsh-autosuggestions zsh-syntax-highlighting z`, set as login shell |
| herdr | herdr + `~/.config/herdr/config.toml` (Ctrl-a prefix, tmux-style keys, dracula) |
| tmux  | tmux + `~/.tmux.conf` (Ctrl-a prefix, `\` / `-` splits, Alt-arrows, mouse, vi copy w/ OSC52) |
| uv    | uv for Python |
| vim   | amix/vimrc basic + `cursorline` + line numbers |

Existing config files are backed up as `*.bak.<timestamp>` before being replaced. Re-running is safe.
