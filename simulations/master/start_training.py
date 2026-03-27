#!/usr/bin/env python3
"""
Human Training Trigger
Usage: python3 start_training.py 5  # 5 rounds
"""
import asyncio
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def trigger_training(rounds: int):
    for attempt in range(5):
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', 8444)
            writer.write(f"START {rounds}\n".encode())
            await writer.drain()
            resp = await reader.readline()
            resp = resp.decode().strip()
            if resp == "OK":
                print(f"✅ Training triggered for {rounds} rounds")
                break
            elif resp == "BUSY":
                print("⚠ Master busy, retrying in 1s...")
                await asyncio.sleep(1)
            else:
                print("❌ Unknown response:", resp)
                break
        except Exception as e:
            print("❌ Failed to connect:", e)
            await asyncio.sleep(1)
        finally:
            if 'writer' in locals():
                writer.close()
                await writer.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger FL Training")
    parser.add_argument("rounds", type=int, help="Number of training rounds")
    args = parser.parse_args()
    asyncio.run(trigger_training(args.rounds))
