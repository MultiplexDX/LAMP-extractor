import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, Form, Response, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from loguru import logger
import traceback
from lamp_extractor import main, utils
from typing import Optional
import base64
from pkg_resources import resource_filename

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# TODO: implementovat index.html pre demovanie
# @app.get("/")
# async def tooth_demo_playground():
#    return FileResponse("src/static/index.html")

#  RnaIntegrity alebo SarsCov2
@app.post("/lamp-extractor/extract/")
async def matrix_extract(
    response: Response,
    rackId: Optional[str] = Form(
        None, example="0123456789", media_type="multipart/form-data"
    ),
    type: Optional[str] = Form(
        None, example="SarsCov2", media_type="multipart/form-data"
    ),
    image: UploadFile = File(..., media_type="multipart/form-data"),
):
    try:
        uploaded_img = load_img(image)
        CONFIG = utils.load_yaml(resource_filename("lamp_extractor", f"apis/rest/config.yaml"))
        matrix, classes, rows, columns, warped_img = main.predict(
            img=uploaded_img,
            config=CONFIG,
            visualize=False
        )

        positions = []
        for col_i, col in enumerate(columns):
            for row_i, row in enumerate(rows):
                class_name = classes[int(matrix[row_i, col_i])]
                # print(class_name)
                position = {"position": f"{row}{col}", "resultType": class_name, "description": ""}
                positions.append(position)
        retval, buffer = cv2.imencode(".jpg", warped_img)
        segmented_plate_base64 = base64.b64encode(buffer)
        response.status_code = status.HTTP_200_OK
        message = {
            "positions": positions,
            "segmentedPlate": segmented_plate_base64,
        }
    except AssertionError as e:
        logger.error(e)
        traceback.print_exc()
        response.status_code = status.HTTP_400_BAD_REQUEST
        message = {"error": str(e)}
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        message = {"error": str(e)}
    finally:
        return message


def load_img(upload_file: UploadFile):
    buffer_file = upload_file.file.read()
    nparr = np.frombuffer(buffer_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

from uvicorn_loguru_integration import run_uvicorn_loguru
if __name__ == "__main__":
    logger.success("Running app...")
    run_uvicorn_loguru(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8081,
            log_level="info",
            reload=True,
        )
    )
