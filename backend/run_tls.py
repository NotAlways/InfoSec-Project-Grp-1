import ssl
import uvicorn

# Create SSL context (server side)
ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

# Load server certificate and private key
ssl_ctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

# Disable old protocols
ssl_ctx.options |= ssl.OP_NO_TLSv1
ssl_ctx.options |= ssl.OP_NO_TLSv1_1
ssl_ctx.options |= ssl.OP_NO_TLSv1_2
# This should set only TLS 1.3 to be used

# Run FastAPI app with this SSL context
uvicorn.run(
    "main:app",      # assuming your FastAPI instance is in main.py as `app`
    host="0.0.0.0",
    port=8443,
    ssl=ssl_ctx
)





# Optional (Manual TLS)
# Load CA that signed client certificates
#ssl_ctx.load_verify_locations(cafile="ca.pem")

# Require client to present a certificate
#ssl_ctx.verify_mode = ssl.CERT_REQUIRED