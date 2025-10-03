import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="POS Service", version="0.1.0")

POS_DATA_PATH = os.getenv("POS_DATA_PATH", "/data/sales.json")


class SaleItem(BaseModel):
    sku: str
    name: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0)


class SaleRequest(BaseModel):
    items: list[SaleItem]
    total: float = Field(ge=0)
    currency: str = "MXN"
    timestamp: datetime | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "pos"}


@app.post("/sales")
def create_sale(sale: SaleRequest):
    computed_total = sum(i.quantity * i.unit_price for i in sale.items)
    # Permitir ligera diferencia por flotantes
    if abs(computed_total - sale.total) > 1e-6:
        return {
            "status": "error",
            "message": "Total inconsistente",
            "computed_total": computed_total,
        }

    sale_id = str(uuid.uuid4())
    record = {
        "id": sale_id,
        "items": [i.model_dump() for i in sale.items],
        "total": sale.total,
        "currency": sale.currency,
        "timestamp": (sale.timestamp or datetime.utcnow()).isoformat(),
    }

    os.makedirs(os.path.dirname(POS_DATA_PATH), exist_ok=True)
    with open(POS_DATA_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"status": "ok", "id": sale_id}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("POS_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)