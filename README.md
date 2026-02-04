# InfoSec-Project-Grp-1

# -LUCAS SECTION-
# To run, u need to install postresql , run CREATE DATABASE notevault;, change the default password to your set password in main.py, then start the backend doing 'cd backend' then 'python -m uvicorn main:app --reload' or 'python -m uvicorn main:app --ssl-certfile cert.pem --ssl-keyfile key.pem' (if you have openssl) in terminal. Then navigate to index.html in your file explorer and open with browser. 
# For key gen, cd backend then run python init_crypto.py

# -PRATHIP SECTION-
# pip install fastapi uvicorn sqlalchemy asyncpg python-multipart passlib[bcrypt] itsdangerous pyotp qrcode[pil] webauthn cryptography pyOpenSSL asn1crypto cbor2 numpy python-dotenv
please install the following packages in VS Code by pasting the above command in the terminal

For normal users: email verification (if new device login) -> TOTP (or backup pin if lost access to Google Auth)

For admin & superadmin users: email verification (if new device login) -> TOTP (or backup pin if lost access to Google Auth) -> Windows Passkey 
