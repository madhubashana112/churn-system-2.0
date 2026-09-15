/* Apply before paint; Light Mode by default, with an explicit saved preference taking priority. */
(function () {
    function saved() { try { return localStorage.getItem('churn-theme') === 'dark' ? 'dark' : 'light'; } catch (_) { return null; } }
    function apply(theme) {
        document.documentElement.dataset.theme = theme;
        document.documentElement.style.colorScheme = theme;
        document.querySelectorAll('[data-theme-toggle]').forEach(button => {
            button.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
            button.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
            button.setAttribute('aria-pressed', String(theme === 'dark'));
        });
        window.dispatchEvent(new Event('themechange'));
    }
    apply(saved() || 'light');
    document.addEventListener('DOMContentLoaded', () => {
        apply(document.documentElement.dataset.theme);
        document.querySelectorAll('[data-theme-toggle]').forEach(button => button.addEventListener('click', () => {
            const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
            try { localStorage.setItem('churn-theme', next); } catch (_) {}
            apply(next);
        }));
    });
    window.addEventListener('storage', e => { if (e.key === 'churn-theme') apply(saved() || 'light'); });
})();
