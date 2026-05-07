#!/usr/bin/env python3
import argparse
import time
import socket, struct
import numpy as np
import cv2
from ultralytics import YOLO

# --- YOLO Setup ---
# Load a small model (YOLOv8n is best for real-time CPU inference)
model = YOLO('yolov8n.pt') 

# Args for setting IP/port of AI-deck.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer with YOLO')
parser.add_argument("-n", default="192.168.0.101", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
        chunk = client_socket.recv(size - len(data))
        if not chunk: break
        data.extend(chunk)
    return data

start = time.time()
count = 0

try:
    while True:
        # Get Packet Info
        packetInfoRaw = rx_bytes(4)
        if not packetInfoRaw: break
        
        [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
        imgHeader = rx_bytes(length - 2)
        [magic, width, height, depth, img_format, size] = struct.unpack('<BHHBBI', imgHeader)

        if magic == 0xBC:
            # Receive the full image stream
            imgStream = bytearray()
            while len(imgStream) < size:
                p_info = rx_bytes(4)
                if not p_info: break
                [p_len, _, _] = struct.unpack('<HBB', p_info)
                chunk = rx_bytes(p_len - 2)
                imgStream.extend(chunk)
            
            # --- Image Conversion ---
            frame = None
            if img_format == 0: # Raw Data
                raw_img = np.frombuffer(imgStream, dtype=np.uint8)
                
                # Check if buffer size matches dimensions to prevent crash
                if len(raw_img) == width * height:
                    raw_img = raw_img.reshape((height, width))
                    
                    # AI-deck raw color is usually Bayer RGGB. 
                    # If the image looks grayscale but with a grid, it's Bayer.
                    # If it's pure grayscale, use COLOR_GRAY2BGR.
                    try:
                        # Attempt Bayer conversion for color
                        frame = cv2.cvtColor(raw_img, cv2.COLOR_BayerRG2BGR)
                    except cv2.error:
                        # Fallback to Grayscale if Bayer fails
                        frame = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"Buffer mismatch: Expected {width*height}, got {len(raw_img)}")
                    continue
            else: # JPEG
                nparr = np.frombuffer(imgStream, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # --- YOLO Inference ---
                results = model.predict(frame, verbose=False, conf=0.5) 
                
                # Annotate the frame with results
                annotated_frame = results[0].plot()

                # --- Metrics & Display ---
                count += 1
                elapsed_time = time.time() - start
                if elapsed_time > 0:
                    fps = count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('AI-Deck YOLO', annotated_frame)
                
                if args.save:
                    cv2.imwrite(f"img_{count:06d}.png", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    client_socket.close()
    cv2.destroyAllWindows()
