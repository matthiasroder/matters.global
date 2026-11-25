"use strict";

// Action tokens arrive only in a fragment. Remove it before making requests
// and retain the value only in this page's closure, never in a form or storage.
(() => {
  let token = new URLSearchParams(window.location.hash.slice(1)).get("token");
  window.history.replaceState(null, "", window.location.pathname);
  const action = document.body.dataset.action;
  const form = document.getElementById("lifecycle-form");
  const submit = document.getElementById("lifecycle-submit");
  const status = document.getElementById("lifecycle-message");
  const password = document.getElementById("reset-password");
  const confirmation = document.getElementById("reset-password-confirm");
  let csrf = null;
  let busy = false;
  let consumed = false;
  submit.disabled = true;

  function clearPasswords() {
    if (password) password.value = "";
    if (confirmation) confirmation.value = "";
  }

  if (!new Set(["verify", "reset"]).has(action) || !token || !/^[A-Za-z0-9_-]{43}$/.test(token)) {
    token = null;
    status.textContent = "This link is incomplete. Request a new email from the Matters account page.";
    form.hidden = true;
    return;
  }

  submit.textContent = action === "verify" ? "Verify email" : "Reset password";
  status.textContent = "Preparing the confirmation form…";

  async function prepare() {
    try {
      const response = await fetch("/api/auth/capabilities", {
        credentials: "same-origin", headers: { Accept: "application/json" }, referrerPolicy: "strict-origin"
      });
      const data = await response.json();
      if (!response.ok || typeof data.csrf_token !== "string" || !data.csrf_token || data.csrf_token.length > 512) throw new Error();
      if (!token || consumed) return;
      csrf = data.csrf_token;
      submit.disabled = false;
      status.textContent = action === "verify"
        ? "Confirm that you want to verify the email address this link was sent to."
        : "Choose a new password. Resetting it signs out every browser and connected client.";
    } catch {
      status.textContent = "The confirmation form is unavailable. Reload this page from your email link to try again.";
    }
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || consumed || !token || !csrf) return;
    const body = { token, csrf_token: csrf };
    if (action === "reset") {
      if (!password || !confirmation || password.value.length < 12 || password.value.length > 1024) {
        status.textContent = "Use between 12 and 1024 characters for the new password.";
        return;
      }
      if (password.value !== confirmation.value) {
        status.textContent = "The new passwords do not match.";
        return;
      }
      body.new_password = password.value;
    }
    busy = true;
    submit.disabled = true;
    clearPasswords();
    status.textContent = "Confirming…";
    try {
      const path = action === "verify" ? "/api/auth/verification/confirm" : "/api/auth/password/reset/confirm";
      const response = await fetch(path, {
        method: "POST", credentials: "same-origin", referrerPolicy: "strict-origin",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      let data;
      try { data = await response.json(); } catch { data = {}; }
      if (!response.ok) {
        status.textContent = data.error?.message || "This link could not be confirmed. Request a new email or try signing in.";
        return;
      }
      const confirmed = action === "verify" ? data.email_verified === true : data.password_changed === true && data.sign_in_required === true;
      if (!confirmed) {
        status.textContent = "The result could not be confirmed. Return to Matters to check your account.";
        return;
      }
      consumed = true;
      token = null;
      csrf = null;
      form.hidden = true;
      status.textContent = action === "verify" ? "Email verified. You can return to Matters." : "Password reset. Sign in with your new password.";
    } catch {
      status.textContent = "The request could not be completed. Try again or return to sign in.";
    } finally {
      busy = false;
      submit.disabled = consumed || !token || !csrf;
    }
  });

  window.addEventListener("pagehide", () => {
    token = null;
    csrf = null;
    clearPasswords();
    submit.disabled = true;
  });
  // Obtaining a CSRF nonce does not verify an email or change a password.
  // Those operations happen only after the user submits the prepared form.
  prepare();
})();
