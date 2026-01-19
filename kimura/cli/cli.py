#!/usr/bin/env python3
import asyncio
import argparse
import sys
import logging
from pathlib import Path
from session.client import PQCClient
from session.server import PQCServer
from session.manager import SessionManager
# Import all keygen functions
from crypto.keygen import (
    generate_mlkem_server_keys, generate_mlkem_client_keys,
    generate_mldsa_server_keys, generate_mldsa_client_keys
)
from protocol.constants import DEFAULT_PORT
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logger = logging.getLogger(__name__)
from protocol.state_machine import StateMachine
async def main():
    parser = argparse.ArgumentParser("pqc-session", description="PQC Secure File Transfer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # ML-KEM KEYGEN COMMANDS
    kem_parser = subparsers.add_parser("mlkem-server", help="Generate server ML-KEM keys")
    kem_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    
    kemc_parser = subparsers.add_parser("mlkem-client", help="Generate client ML-KEM keys")
    kemc_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    
    #  ML-DSA KEYGEN COMMANDS  
    dsas_parser = subparsers.add_parser("mldsa-server", help="Generate server ML-DSA keys")
    dsas_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    
    dsac_parser = subparsers.add_parser("mldsa-client", help="Generate client ML-DSA keys")
    dsac_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    
    #  TRANSFER COMMANDS (unchanged)
    transfer_parser = subparsers.add_parser("transfer", help="PQC file transfer")
    transfer_subparsers = transfer_parser.add_subparsers(dest="mode", help="Transfer mode")
    
    # SERVER MODE
    server_parser = transfer_subparsers.add_parser("server", help="Receive file")
    server_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    server_parser.add_argument("--output", required=True, help="Output file path")
    server_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    server_parser.add_argument("--persistent", action="store_true", 
                          help="Keep server alive for multiple GPU clients")
    # CLIENT MODE  
    client_parser = transfer_subparsers.add_parser("client", help="Send file")
    client_parser.add_argument("host", help="server:port (ex: 127.0.0.1:8443)")
    client_parser.add_argument("--file", required=True, help="File to send")
    client_parser.add_argument("--key-path", default="./keys", help="Path to keys/")
    
    args = parser.parse_args()
    
    # REPLACE keygen calls with ABSOLUTE PATHS:
    if args.command == "mlkem-server":
        generate_mlkem_server_keys(args.key_path)
        return
    elif args.command == "mlkem-client":
        generate_mlkem_client_keys(args.key_path)
        return
    elif args.command == "mldsa-server":
        generate_mldsa_server_keys(args.key_path)
        return
    elif args.command == "mldsa-client":
        generate_mldsa_client_keys(args.key_path)
        return


    elif args.command == "transfer":
        if args.mode == "server":
            if args.persistent:
                # FIXED: Use SINGLE PQCServer instance
                server = PQCServer(args.key_path, args.output)
                await server.serve_forever(port=args.port)

            else:
                # Single client fallback
                mgr = SessionManager("server", args.key_path, args.output)
                await mgr.establish_channel()
                await mgr.recv_file(args.output)
                await mgr.close()
                logger.info(f"File verified: {args.output}")

        elif args.mode == "client":
            if hasattr(args, 'receive') and args.receive:
                client = PQCClient(args.key_path)  # No file needed
                await client.connect_fl(args.host.split(':')[0], int(args.host.split(':')[1]))
                data = await client.recv_data()    # RECEIVE from server!
                with open("received_from_server.bin", "wb") as f:
                    f.write(data)
            else:
                client = PQCClient(args.key_path, args.file)
                host_port = args.host.split(":")
                await client.connect_and_send(host_port[0], int(host_port[1]))
                logger.info(f"File sent: {args.file}")

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown")
