#!/usr/bin/env bash
# oh-my-server: bootstrap a fresh server with my favorite tools.
#
# Usage:
#   bash setup.sh              # interactive: pick what to install
#   bash setup.sh --all        # install everything, no questions
#   curl -fsSL https://raw.githubusercontent.com/zlwang-cs/oh-my-server/main/setup.sh | bash
set -euo pipefail

# ---------------------------------------------------------------- helpers ---
c_info=$'\033[1;34m'; c_ok=$'\033[1;32m'; c_warn=$'\033[1;33m'; c_err=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$c_info" "$c_off" "$*"; }
ok()   { printf '%s ✓ %s %s\n' "$c_ok" "$c_off" "$*"; }
warn() { printf '%s !  %s %s\n' "$c_warn" "$c_off" "$*" >&2; }
die()  { printf '%s ✗  %s %s\n' "$c_err" "$c_off" "$*" >&2; exit 1; }
has()  { command -v "$1" >/dev/null 2>&1; }

STAMP="$(date +%Y%m%d-%H%M%S)"

# Back up a file before overwriting it (only if it exists and differs).
backup() {
    local f="$1"
    if [ -e "$f" ] || [ -L "$f" ]; then
        cp -a "$f" "$f.bak.$STAMP"
        warn "backed up existing $f -> $f.bak.$STAMP"
    fi
}

# Write stdin to $1, backing up the old file only when content changes.
write_config() {
    local dest="$1" tmp
    tmp="$(mktemp)"
    cat >"$tmp"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ] && cmp -s "$tmp" "$dest"; then
        rm -f "$tmp"; ok "$dest already up to date"; return
    fi
    backup "$dest"
    mv "$tmp" "$dest"
    ok "wrote $dest"
}

# Append a line to a file unless it is already there.
append_once() {
    local line="$1" file="$2"
    touch "$file"
    grep -qxF -- "$line" "$file" || printf '%s\n' "$line" >>"$file"
}

# ------------------------------------------------------- package manager ---
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if has sudo; then SUDO="sudo"; else SUDO="__none__"; fi
fi

PM=""
if   has apt-get; then PM=apt
elif has dnf;     then PM=dnf
elif has yum;     then PM=yum
elif has pacman;  then PM=pacman
elif has zypper;  then PM=zypper
elif has apk;     then PM=apk
elif has brew;    then PM=brew
fi

APT_UPDATED=0
pkg_install() {
    [ $# -eq 0 ] && return 0
    [ -n "$PM" ] || { warn "no supported package manager; please install manually: $*"; return 1; }
    local s="$SUDO"
    if [ "$PM" = brew ]; then s=""
    elif [ "$s" = "__none__" ]; then warn "not root and no sudo; please install manually: $*"; return 1
    fi
    log "installing packages: $*"
    case "$PM" in
        apt)
            if [ "$APT_UPDATED" -eq 0 ]; then $s apt-get update -y; APT_UPDATED=1; fi
            $s env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@" ;;
        dnf)    $s dnf install -y "$@" ;;
        yum)    $s yum install -y "$@" ;;
        pacman) $s pacman -Sy --noconfirm --needed "$@" ;;
        zypper) $s zypper --non-interactive install "$@" ;;
        apk)    $s apk add --no-cache "$@" ;;
        brew)   brew install "$@" ;;
    esac
}

# Install each command's package only if the command is missing.
ensure() {
    local missing=() c
    for c in "$@"; do has "$c" || missing+=("$c"); done
    [ ${#missing[@]} -eq 0 ] || pkg_install "${missing[@]}"
}

# ----------------------------------------------------------------- herdr ---
install_herdr() {
    ensure curl
    if has herdr || [ -x "$HOME/.local/bin/herdr" ]; then
        ok "herdr already installed"
    else
        log "installing herdr"
        curl -fsSL https://herdr.dev/install.sh | sh
    fi

    log "writing herdr config"
    write_config "$HOME/.config/herdr/config.toml" <<'EOF'
onboarding = false

[ui.toast]
delivery = "herdr"

# --- translated from ~/.tmux.conf ---
[keys]
prefix = "ctrl+a"                                   # tmux: set -g prefix C-a
split_vertical   = ["prefix+backslash", "prefix+v"] # tmux: bind \ split-window -h (side by side)
split_horizontal = "prefix+minus"                   # tmux: bind - split-window -v (stacked; same as Herdr default)
focus_pane_left  = ["prefix+h", "alt+left"]         # tmux: bind -n M-Left  select-pane -L
focus_pane_right = ["prefix+l", "alt+right"]        # tmux: bind -n M-Right select-pane -R
focus_pane_up    = ["prefix+k", "alt+up"]           # tmux: bind -n M-Up    select-pane -U
focus_pane_down  = ["prefix+j", "alt+down"]         # tmux: bind -n M-Down  select-pane -D

[theme]
name = "dracula"
auto_switch = false

[ui]
status_indicators = "dots"
EOF
}

# ------------------------------------------------------------------ tmux ---
# Based on https://raw.githubusercontent.com/zlwang-cs/ServerUtils/main/code/tmux.conf
# with fixes:
#   * `bind \\` -> `bind '\'`: a quoted backslash parses the same on every
#     tmux version, and survives copy/paste through shells (an unquoted
#     `bind \ split...` silently binds the wrong key on tmux >= 3.0).
#   * The old `y` binding piped to `printf "\033]52..."`, but copy-pipe's
#     stdout is not the terminal, so the OSC52 sequence never reached the
#     client. Use tmux's built-in set-clipboard + Ms override instead.
#   * `setw -g mouse` -> `set -g mouse` (mouse is a session option).
install_tmux() {
    ensure tmux
    log "writing tmux config"
    write_config "$HOME/.tmux.conf" <<'EOF'
# ---- prefix: Ctrl-a, splits, reload ----
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

bind '\' split-window -h
bind - split-window -v
unbind '"'
unbind %

bind r source-file ~/.tmux.conf \; display-message 'tmux.conf reloaded'

bind -n M-Left  select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up    select-pane -U
bind -n M-Down  select-pane -D

# ---- mouse & copy mode ----
set -g mouse on
setw -g mode-keys vi

# y: copy to the system clipboard via OSC52 (works over SSH -> local Mac/PC).
# Enter copy mode: Ctrl-a [   select: v / V / Ctrl-v   copy: y
set -g set-clipboard on
set -as terminal-overrides ',*:Ms=\E]52;%p1%s;%p2%s\007'
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi V send -X select-line
bind -T copy-mode-vi C-v send -X rectangle-toggle
bind -T copy-mode-vi y send -X copy-selection-and-cancel

# prefix + m: toggle mouse (turn off to use the terminal's native selection)
bind m set -g mouse \; display-message 'mouse: #{?mouse,on,off}'

# ---- terminal & colors (true color) ----
set -g default-terminal "xterm-256color"
set -as terminal-overrides ',*:Tc'
EOF
}

# -------------------------------------------------------------------- uv ---
install_uv() {
    ensure curl
    if has uv || [ -x "$HOME/.local/bin/uv" ]; then
        ok "uv already installed"
    else
        log "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

# ------------------------------------------------------ zsh + oh-my-zsh ---
install_zsh() {
    ensure zsh git curl
    has zsh || die "zsh is not available; cannot set up oh-my-zsh"

    local zsh_dir="${ZSH:-$HOME/.oh-my-zsh}"
    if [ -d "$zsh_dir" ]; then
        ok "oh-my-zsh already installed"
    else
        log "installing oh-my-zsh"
        RUNZSH=no CHSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    fi

    local custom="${ZSH_CUSTOM:-$zsh_dir/custom}/plugins" p
    for p in zsh-autosuggestions zsh-syntax-highlighting; do
        if [ -d "$custom/$p" ]; then
            ok "plugin $p already present"
        else
            log "cloning plugin $p"
            git clone --depth=1 "https://github.com/zsh-users/$p" "$custom/$p"
        fi
    done

    # Replace the (single- or multi-line) plugins=(...) block in ~/.zshrc.
    local rc="$HOME/.zshrc"
    [ -f "$rc" ] || cp "$zsh_dir/templates/zshrc.zsh-template" "$rc"
    backup "$rc"
    awk '
        BEGIN { done = 0; skip = 0 }
        !done && /^[[:space:]]*plugins=\(/ {
            print "plugins=("
            print "  git"
            print "  zsh-autosuggestions"
            print "  zsh-syntax-highlighting"
            print "  z"
            print ")"
            done = 1
            if ($0 !~ /\)/) skip = 1
            next
        }
        skip { if ($0 ~ /\)/) skip = 0; next }
        { print }
        END { if (!done) { print "plugins=(git zsh-autosuggestions zsh-syntax-highlighting z)" } }
    ' "$rc" >"$rc.tmp" && mv "$rc.tmp" "$rc"
    ok "set oh-my-zsh plugins in $rc"

    # herdr and uv live in ~/.local/bin.
    append_once 'export PATH="$HOME/.local/bin:$PATH"' "$rc"

    # Make zsh the login shell.
    local zsh_path; zsh_path="$(command -v zsh)"
    if [ "${SHELL:-}" = "$zsh_path" ]; then
        ok "zsh is already the default shell"
    elif chsh -s "$zsh_path" </dev/tty 2>/dev/null || { [ "$SUDO" != "__none__" ] && $SUDO chsh -s "$zsh_path" "$(id -un)"; }; then
        ok "default shell changed to zsh (takes effect on next login)"
    else
        warn "could not change the default shell; run: chsh -s $zsh_path"
    fi
}

# ------------------------------------------------------------------- vim ---
install_vim() {
    ensure vim git
    if [ -d "$HOME/.vim_runtime/.git" ]; then
        log "updating amix/vimrc"
        git -C "$HOME/.vim_runtime" pull --ff-only || warn "could not update ~/.vim_runtime"
    else
        log "installing amix/vimrc (basic)"
        git clone --depth=1 https://github.com/amix/vimrc.git "$HOME/.vim_runtime"
    fi
    sh "$HOME/.vim_runtime/install_basic_vimrc.sh"
    append_once "set cursorline" "$HOME/.vimrc"
    append_once "set number" "$HOME/.vimrc"
    ok "vim configured"
}

# ------------------------------------------------------------------- menu ---
# Order matters: zsh first so oh-my-zsh's fresh ~/.zshrc doesn't clobber
# PATH lines that the uv/herdr installers add.
APPS=(zsh herdr tmux uv vim)
declare -A DESC=(
    [zsh]="zsh + oh-my-zsh (plugins: git, zsh-autosuggestions, zsh-syntax-highlighting, z)"
    [herdr]="herdr + config (~/.config/herdr/config.toml)"
    [tmux]="tmux + config (~/.tmux.conf)"
    [uv]="uv (Python package/project manager)"
    [vim]="vim + amix/vimrc basic + cursorline + line numbers"
)
declare -A PICK=()

choose_apps() {
    # Read from the terminal even when the script itself is piped into bash.
    local tty=/dev/tty
    { : <"$tty"; } 2>/dev/null || die "no terminal to ask on; rerun with --all"

    echo
    echo "Which apps do you want to install?"
    local i=1 a
    for a in "${APPS[@]}"; do printf '  %d) %s\n' "$i" "${DESC[$a]}"; i=$((i + 1)); done
    echo
    local ans
    read -r -p "Enter numbers separated by spaces, or 'a' for all [a]: " ans <"$tty"
    ans="${ans:-a}"
    if [[ "$ans" =~ ^[aA] ]]; then
        for a in "${APPS[@]}"; do PICK[$a]=1; done
    else
        local n
        for n in ${ans//,/ }; do
            [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#APPS[@]} ] \
                || die "invalid choice: $n"
            PICK[${APPS[$((n - 1))]}]=1
        done
    fi

    echo
    echo "Will install:"
    for a in "${APPS[@]}"; do [ -n "${PICK[$a]:-}" ] && echo "  - ${DESC[$a]}"; done
    read -r -p "Proceed? [Y/n] " ans <"$tty"
    [[ "${ans:-y}" =~ ^[yY] ]] || die "aborted"
}

main() {
    case "${1:-}" in
        --all|-a|-y) for a in "${APPS[@]}"; do PICK[$a]=1; done ;;
        -h|--help)   sed -n '2,8p' "$0"; exit 0 ;;
        "")          choose_apps ;;
        *)           die "unknown option: $1 (try --help)" ;;
    esac

    local a failed=()
    for a in "${APPS[@]}"; do
        [ -n "${PICK[$a]:-}" ] || continue
        echo; log "---- $a ----"
        # Run each app in a subshell so one failure doesn't stop the rest.
        if ! ( "install_$a" ); then failed+=("$a"); warn "$a failed"; fi
    done

    echo
    if [ ${#failed[@]} -eq 0 ]; then
        ok "all done! start a new shell (or run: exec zsh) to pick up changes"
    else
        die "finished with failures: ${failed[*]}"
    fi
}

main "$@"
