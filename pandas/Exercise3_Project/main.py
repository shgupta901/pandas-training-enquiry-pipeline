from typing import Optional
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

import models
import schemas
from database import Base, engine, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Consultant and Practice Area API")


@app.post(
    "/practice-areas",
    response_model=schemas.PracticeAreaOut,
    status_code=status.HTTP_201_CREATED,
)
def create_practice_area(
    payload: schemas.PracticeAreaCreate,
    db: Session = Depends(get_db),
):
    practice_area = models.PracticeArea(
        name=payload.name,
        floor_number=payload.floor_number,
    )
    db.add(practice_area)
    db.commit()
    db.refresh(practice_area)
    return practice_area


@app.post(
    "/practice-areas/{parent_id}/consultants",
    response_model=schemas.ConsultantOut,
    status_code=status.HTTP_201_CREATED,
)
def create_consultant(
    parent_id: int,
    payload: schemas.ConsultantCreate,
    db: Session = Depends(get_db),
):
    parent = (
        db.query(models.PracticeArea)
        .filter(models.PracticeArea.id == parent_id)
        .first()
    )
    if not parent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice Area not found",
        )

    consultant = models.Consultant(
        name=payload.name,
        designation=payload.designation,
        consultation_fee=payload.consultation_fee,
        experience_years=payload.experience_years,
        practicearea_id=parent_id,
    )
    db.add(consultant)
    db.commit()
    db.refresh(consultant)
    return consultant


@app.get(
    "/practice-areas/{parent_id}",
    response_model=schemas.PracticeAreaWithConsultants,
    status_code=status.HTTP_200_OK,
)
def get_practice_area(
    parent_id: int,
    designation: Optional[str] = None,
    sort: Optional[str] = None,
    db: Session = Depends(get_db),
):
    parent = (
        db.query(models.PracticeArea)
        .filter(models.PracticeArea.id == parent_id)
        .first()
    )
    if not parent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice Area not found",
        )

    query = db.query(models.Consultant).filter(
        models.Consultant.practicearea_id == parent_id
    )

    # Apply exact match filter if designation is supplied and non-blank
    if designation is not None and designation.strip() != "":
        query = query.filter(models.Consultant.designation == designation)

    # Validate and apply ascending sort if sort parameter is provided
    if sort is not None:
        if sort == "name":
            query = query.order_by(models.Consultant.name.asc())
        elif sort == "consultation_fee":
            query = query.order_by(models.Consultant.consultation_fee.asc())
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid sort field. Allowed values: name, consultation_fee",
            )

    consultants = query.all()

    return schemas.PracticeAreaWithConsultants(
        id=parent.id,
        name=parent.name,
        floor_number=parent.floor_number,
        consultants=consultants,
    )


@app.put(
    "/practice-areas/{parent_id}/consultants/{child_id}",
    response_model=schemas.ConsultantOut,
    status_code=status.HTTP_200_OK,
)
def update_consultant(
    parent_id: int,
    child_id: int,
    payload: schemas.ConsultantCreate,
    db: Session = Depends(get_db),
):
    parent = (
        db.query(models.PracticeArea)
        .filter(models.PracticeArea.id == parent_id)
        .first()
    )
    if not parent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice Area not found",
        )

    consultant = (
        db.query(models.Consultant)
        .filter(
            models.Consultant.id == child_id,
            models.Consultant.practicearea_id == parent_id,
        )
        .first()
    )
    if not consultant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultant not found under this Practice Area",
        )

    consultant.name = payload.name
    consultant.designation = payload.designation
    consultant.consultation_fee = payload.consultation_fee
    consultant.experience_years = payload.experience_years

    db.commit()
    db.refresh(consultant)
    return consultant


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)