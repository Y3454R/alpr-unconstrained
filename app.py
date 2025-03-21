from fastapi import FastAPI, File, UploadFile
import aiofiles
import os
from pypeline import vehicle_detection  # Your existing detection logic
import asyncio

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/alpr/detect")
async def detect_license_plate(image: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, image.filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        while content := await image.read(1024):  # Read in chunks
            await out_file.write(content)

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, vehicle_detection, file_path)

    # print(f"resres: {results}")

    os.remove(file_path)  # Clean up after processing
    return {"message": "Detection successful", "data": results}
