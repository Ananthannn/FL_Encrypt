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
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', 8444)
        cmd = f"START {rounds}\n".encode()
        writer.write(cmd)
        await writer.drain()
        logger.info(f"Triggered {rounds} training rounds! ")
    except Exception as e:
        logger.error(f"❌ Failed to trigger: {e}")
    finally:
        if 'writer' in locals():
            writer.close()
            await writer.wait_closed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger FL Training")
    parser.add_argument("rounds", type=int, help="Number of training rounds")
    args = parser.parse_args()
    asyncio.run(trigger_training(args.rounds))
