from firebase_admin.credentials import Certificate
import pytest
import os
import base64
import json
from symbiont.fb import init_firebase
import firebase_admin
from unittest.mock import patch


def test_missing_firebase_credentials(monkeypatch):
    monkeypatch.delenv("FIREBASE_CREDENTIALS", raising=False)
    with pytest.raises(KeyError):
        _ = os.environ["FIREBASE_CREDENTIALS"]


def test_init_firebase(monkeypatch):
    pass

    # # Mock environment variables
    credentials = {
        "type": "service_account",
        "project_id": "my-project-id",
        "private_key_id": "some-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk@my-project-id.iam.gserviceaccount.com",
        "client_id": "some-client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk%40my-project-id.iam.gserviceaccount.com",
    }

    mock_cred = json.dumps(credentials)

    with patch.object(firebase_admin, "initialize_app") as mock_initialize_app, patch.object(
        firebase_admin, "credentials", return_value=mock_cred
    ) as mock_credentials:
        cred = mock_credentials.Certificate.return_value
        assert cred is not None
        mock_initialize_app(cred, {"storageBucket": "my-bucket"})
        init_firebase()
        mock_initialize_app.assert_called_once_with(cred, {"storageBucket": "my-bucket"})
