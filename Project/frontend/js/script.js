var API_BASE = window.API_BASE || "http://127.0.0.1:5000";

document.addEventListener('DOMContentLoaded', function() {
    initTooltips();
    initContactForm();
    initAutoDismissAlerts();
    initSmoothScroll();
});

function initTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(el) {
        return new bootstrap.Tooltip(el);
    });
}

function initContactForm() {
    var form = document.getElementById('contactForm');
    if (!form) return;

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        var feedback = document.getElementById('formFeedback');
        var btn = form.querySelector('button[type="submit"]');
        var originalText = btn.innerHTML;

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Sending...';

        setTimeout(function() {
            feedback.innerHTML = '<div class="alert alert-success alert-dismissible fade show" role="alert">' +
                '<i class="bi bi-check-circle-fill"></i> Thank you! Your message has been sent successfully. We will get back to you soon.' +
                '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>' +
                '</div>';
            btn.disabled = false;
            btn.innerHTML = originalText;
            form.reset();
        }, 1500);
    });
}

function initAutoDismissAlerts() {
    var alerts = document.querySelectorAll('.alert-dismissible:not(.persistent)');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
}

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

function validateForm(formId) {
    var form = document.getElementById(formId);
    if (!form) return true;
    var inputs = form.querySelectorAll('input[required], textarea[required], select[required]');
    var valid = true;

    inputs.forEach(function(input) {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            valid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return valid;
}

function animateCounter(el, target) {
    var current = 0;
    var step = Math.ceil(target / 50);
    var timer = setInterval(function() {
        current += step;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        el.textContent = current.toLocaleString();
    }, 30);
}
