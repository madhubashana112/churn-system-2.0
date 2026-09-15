(function () {
    const form = document.getElementById('auth-form');
    function clearWorkspace() {
        ['tenant_id', 'tenant_name', 'tenant_sector'].forEach(key => localStorage.removeItem(key));
    }
    if (form) {
        document.getElementById('show-password').addEventListener('change', event => {
            ['password', 'confirm-password'].forEach(id => {
                const input = document.getElementById(id);
                if (input) input.type = event.target.checked ? 'text' : 'password';
            });
        });
        form.addEventListener('submit', async event => {
            event.preventDefault();
            const error = document.getElementById('auth-error');
            const button = document.getElementById('auth-submit');
            const label = button.textContent;
            error.hidden = true;
            const payload = Object.fromEntries(new FormData(form));
            const confirm = document.getElementById('confirm-password');
            if (confirm && confirm.value !== payload.password) {
                error.textContent = 'Passwords do not match.'; error.hidden = false; return;
            }
            button.disabled = true; button.textContent = 'Please wait…';
            try {
                const response = await fetch('/api/auth/' + form.dataset.mode, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
                const data = await response.json();
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check your details and try again.');
                clearWorkspace();
                window.location.assign(data.redirect);
            } catch (err) { error.textContent = err.message; error.hidden = false; }
            finally { button.disabled = false; button.textContent = label; }
        });
    }
    document.querySelectorAll('[data-logout]').forEach(button => button.addEventListener('click', async () => {
        button.disabled = true;
        try {
            const response = await fetch('/api/auth/logout', {method:'POST'});
            if (!response.ok) throw new Error('Log out failed. Please try again.');
            clearWorkspace(); window.location.assign('/login');
        } catch (err) { button.disabled = false; alert(err.message); }
    }));
})();
