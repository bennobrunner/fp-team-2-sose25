from sanic import Sanic, json

app = Sanic("TestApp")

@app.get("/moin")
async def moin(request):
    return json({"message": "Moin, moin!"})

@app.post("/fingers")
async def fingers(request):
    data = request.json()
    landmarks = data.get("landmarks")
    print(landmarks)


def __main__():
    app.run(host="localhost", port=8000)
if __name__ == "__main__":
    __main__()
