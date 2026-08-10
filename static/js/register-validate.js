document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const emailInput = document.getElementById("email");
    const emailBox = emailInput.closest(".input-box");
    const emailError = document.getElementById("email-error");

    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    function validateEmail() {
        const value = emailInput.value.trim();

        if (value === "" || !emailPattern.test(value)) {
            emailBox.classList.add("invalid");
            return false;
        } else {
            emailBox.classList.remove("invalid");
            return true;
        }
    }

    // Real-time validation as user types (after first attempt)
    emailInput.addEventListener("input", function () {
        if (emailBox.classList.contains("invalid")) {
            validateEmail();
        }
    });

    // Validate on blur (when user leaves the field)
    emailInput.addEventListener("blur", validateEmail);

    // Validate on form submit — block submission if invalid
    form.addEventListener("submit", function (e) {
        const isEmailValid = validateEmail();

        if (!isEmailValid) {
            e.preventDefault();
            emailInput.focus();
        }
    });

});
