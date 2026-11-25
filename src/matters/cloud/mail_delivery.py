"""Explicit TLS mail delivery; no default sender or credential-bearing outbox."""

from dataclasses import dataclass
from email.message import EmailMessage
import json
import os
from pathlib import Path
import smtplib
import socket
import ssl
import stat
import sys
import threading

from .accounts import _normalized_email


DELIVERY_OUTCOMES = ("requested", "queued", "sent", "failed", "dropped", "skipped", "closed")
MAX_DELIVERY_COUNT = 2**63 - 1


class JSONMailEventSink:
    """Opt-in operator JSON lines with a fixed, secret-free event schema.

    This owns no public endpoint and changes no global logging configuration.
    The process/container log collector supplies worker identity and retention.
    """

    def __init__(self, stream=None):
        self.stream = sys.stderr if stream is None else stream
        self._lock = threading.Lock()

    def __call__(self, event):
        if not isinstance(event, dict) or event.get("outcome") not in DELIVERY_OUTCOMES or type(event.get("count")) is not int or not 0 <= event["count"] <= MAX_DELIVERY_COUNT:
            raise ValueError("Invalid mail delivery event.")
        # Never forward arbitrary fields or text from callers/providers.
        payload = {"event": "matters_mail_delivery", "outcome": event["outcome"], "count": event["count"]}
        with self._lock:
            self.stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.stream.flush()


class MailDeliveryError(RuntimeError):
    """A safe operator error that never includes message bodies or credentials."""


@dataclass(frozen=True, repr=False)
class MailMessage:
    recipient: str
    subject: str
    body: str


class DisabledMailDelivery:
    configured = False

    def send(self, recipient, subject, body):
        raise MailDeliveryError("Email delivery is not configured.")


class CaptureMailDelivery:
    """An explicitly injected in-memory test sink, never selected by config."""

    configured = True
    test_only = True

    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()

    def send(self, recipient, subject, body):
        with self._lock:
            self.messages.append(MailMessage(recipient, subject, body))


def _private_secret(path):
    try:
        # Refuse FIFOs/devices after fstat without blocking while opening one.
        descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {0, os.geteuid()}:
                raise ValueError()
            data = stream.read(4097)
        if not 1 <= len(data) <= 4096:
            raise ValueError()
        password = data.decode("utf-8").rstrip("\r\n")
        if not password or "\x00" in password or "\r" in password or "\n" in password:
            raise ValueError()
        return password
    except (OSError, ValueError, UnicodeError):
        raise MailDeliveryError("The SMTP password file must be private and readable.") from None


class SMTPMailDelivery:
    """SMTP with verified TLS, optional login, and a private password file."""

    configured = True

    def __init__(self, host, port, sender, *, username=None, password_file=None, tls="starttls", ca_file=None, timeout=10):
        if not isinstance(host, str) or not host or len(host) > 253 or any(c.isspace() or ord(c) < 32 for c in host) or "/" in host:
            raise MailDeliveryError("The SMTP host is invalid.")
        if type(port) is not int or not 1 <= port <= 65535 or tls not in {"starttls", "implicit"}:
            raise MailDeliveryError("SMTP requires a valid port and verified TLS.")
        if type(timeout) not in (int, float) or not 0 < timeout <= 10:
            raise MailDeliveryError("The SMTP timeout is invalid.")
        if bool(username) != bool(password_file) or (username is not None and (not isinstance(username, str) or len(username) > 320 or any(ord(c) < 32 for c in username))):
            raise MailDeliveryError("Configure both SMTP username and private password file.")
        try:
            sender = _normalized_email(sender)
            context = ssl.create_default_context(cafile=ca_file)
        except Exception:
            raise MailDeliveryError("The SMTP sender or TLS trust configuration is invalid.") from None
        if password_file:
            _private_secret(password_file)
        self.host, self.port, self.sender = host, port, sender
        self.username, self.password_file = username, password_file
        self.tls, self.context, self.timeout = tls, context, timeout
        self._connections, self._lock, self._closed = set(), threading.Lock(), False

    @classmethod
    def from_env(cls, environ=None):
        return mail_delivery_from_env(environ)

    def send(self, recipient, subject, body):
        connection = None
        try:
            with self._lock:
                if self._closed:
                    raise MailDeliveryError("Email delivery is closed.")
            recipient = _normalized_email(recipient)
            if not isinstance(subject, str) or not 1 <= len(subject) <= 200 or "\r" in subject or "\n" in subject or not isinstance(body, str) or len(body) > 32768:
                raise ValueError()
            message = EmailMessage()
            message["From"], message["To"], message["Subject"] = self.sender, recipient, subject
            message.set_content(body)
            if self.tls == "implicit":
                connection = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=self.context)
            else:
                connection = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            with self._lock:
                if self._closed:
                    connection.close()
                    raise MailDeliveryError("Email delivery is closed.")
                self._connections.add(connection)
            with connection:
                connection.ehlo()
                if self.tls == "starttls":
                    connection.starttls(context=self.context)
                    connection.ehlo()
                if self.username:
                    connection.login(self.username, _private_secret(self.password_file))
                connection.send_message(message)
        except Exception:
            # An SMTP response, URL, recipient or library exception must never
            # become public error text or a token-bearing application log.
            raise MailDeliveryError("Email delivery could not be completed.") from None
        finally:
            if connection is not None:
                with self._lock:
                    self._connections.discard(connection)

    def close(self):
        with self._lock:
            self._closed = True
            active = list(self._connections)
        for connection in active:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            try:
                connection.close()
            except OSError:
                pass


def mail_delivery_from_env(environ=None):
    environ = os.environ if environ is None else environ
    host = environ.get("MATTERS_SMTP_HOST")
    names = {"MATTERS_SMTP_PORT", "MATTERS_SMTP_SENDER", "MATTERS_SMTP_USERNAME", "MATTERS_SMTP_PASSWORD_FILE", "MATTERS_SMTP_TLS", "MATTERS_SMTP_CA_FILE"}
    if not host:
        if any(environ.get(name) for name in names):
            raise MailDeliveryError("SMTP configuration is incomplete.")
        return DisabledMailDelivery()
    try:
        tls = environ.get("MATTERS_SMTP_TLS", "starttls")
        port = int(environ.get("MATTERS_SMTP_PORT", "465" if tls == "implicit" else "587"))
    except ValueError:
        raise MailDeliveryError("The SMTP port is invalid.") from None
    return SMTPMailDelivery(host, port, environ.get("MATTERS_SMTP_SENDER"), username=environ.get("MATTERS_SMTP_USERNAME"), password_file=environ.get("MATTERS_SMTP_PASSWORD_FILE"), tls=tls, ca_file=environ.get("MATTERS_SMTP_CA_FILE"))
