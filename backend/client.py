import cv2
import base64
import asyncio
import websockets
import numpy as np

SERVER_URI = "ws://10.21.124.185:8000/ws"  # your server's IP + WebSocket route

async def receive_video():
    async with websockets.connect(SERVER_URI) as websocket:  # you missed passing the URI
        print("✅ Connected to video stream")
        try:
            while True:
                data = await websocket.recv()
                jpg_bytes = base64.b64decode(data)
                img_array = np.frombuffer(jpg_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                cv2.imshow("Remote Camera Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed by server.")
        finally:
            cv2.destroyAllWindows()

asyncio.run(receive_video())
