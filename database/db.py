import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    url = os.getenv("DATABASE_URL")
    print("Using DB:", url)

    return psycopg2.connect(
        dsn=url,
        cursor_factory=RealDictCursor
    )


def create_tables():

    conn = get_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        fullname TEXT NOT NULL,

        email TEXT UNIQUE NOT NULL,

        password TEXT NOT NULL,

        age INTEGER,

        gender TEXT,

        weight REAL,

        blood_group TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    # Prediction History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id INTEGER,

        age INTEGER,

        gender TEXT,

        weight REAL,

        symptoms TEXT,

        disease TEXT,

        confidence REAL,

        severity TEXT,

        doctor TEXT,

        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(user_id) REFERENCES users(id)

    )
    """)

    # Appointments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id INTEGER,

        doctor TEXT,

        hospital TEXT,

        appointment_date TEXT,

        appointment_time TEXT,

        status TEXT DEFAULT 'Pending',

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(user_id) REFERENCES users(id)

    )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    user_id,
    age,
    gender,
    weight,
    symptoms,
    disease,
    confidence,
    severity,
    doctor
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (
            user_id,
            age,
            gender,
            weight,
            symptoms,
            disease,
            confidence,
            severity,
            doctor
        )

      VALUES
(
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s
)
        """,
        (
            user_id,
            age,
            gender,
            weight,
            symptoms,
            disease,
            confidence,
            severity,
            doctor
        )
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    print("Database Created Successfully")