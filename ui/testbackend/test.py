#!/usr/bin/env python

import asyncio
import websockets
import os
import json

async def echo(websocket):  # Removed 'path' parameter as it's no longer needed in newer websockets versions
    try:
        async for message in websocket:
            print("Received message:", message, flush=True)
            
            # Echo the message back
            await websocket.send(message)
            await websocket.send("[END]")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

async def main():
    print("WebSocket server starting", flush=True)
    
    # Create the server with CORS headers
    async with websockets.serve(
        echo,
        "0.0.0.0",
        int(os.environ.get('PORT', 8090))
    ) as server:
        print("WebSocket server running on port 8090", flush=True)
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())