import os
import asyncio
import aiofiles
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from pypeline import vehicle_detection

app = FastAPI()
UPLOAD_DIR = "uploads"


@app.post("/alpr/detect")
async def detect_license_plate(
    image: UploadFile = File(...), fileName: str = Form(...)
):
    file_path = os.path.join(UPLOAD_DIR, image.filename)

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        print(fileName)

        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await image.read(1024):
                await out_file.write(content)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, vehicle_detection, file_path)
        # results = ["demo"]

        return {"message": "success", "fileName": fileName, "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up file: {cleanup_error}")
