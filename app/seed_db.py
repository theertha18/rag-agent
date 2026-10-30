"""
Creates the small SQLite database the agent's `find_projects` tool reads.

Run with:  python -m app.seed_db

The rows come from the CV and work notes. Re-running rebuilds the table from
scratch, so editing PROJECTS below and re-running is the way to update it.
"""

import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).parent.parent / "data" / "portfolio.db"

# (name, kind, description, technologies)
PROJECTS = [
    (
        "Master's thesis: infant carrier seat occupancy detection",
        "academic",
        "Detects whether a baby is in a car seat using ultrasonic sensing and machine learning. "
        "Signal cleaning, feature extraction and selection, an Isolation Forest safety gate, six "
        "classifiers compared (Random Forest chosen, 98% cross-validation accuracy), SHAP "
        "explanations and a Streamlit GUI.",
        "Python, NumPy, Scikit-learn, TensorFlow, SHAP, Streamlit, ultrasonic sensors",
    ),
    (
        "Service Management Portal",
        "academic",
        "Component of a provider management platform for creating service requests, tracking their "
        "status and evaluating provider offers. Built in an Agile/Kanban team, deployed on AWS.",
        "Python, Django, PostgreSQL, Spring Boot, REST API, AWS, Agile, Kanban",
    ),
    (
        "Cloud-based anomaly detection system",
        "academic",
        "Learns number sequences with Hierarchical Temporal Memory and flags anomalies. Packaged as a "
        "Docker container on Azure, triggered through an Azure Storage Queue.",
        "C#, NeoCortexApi, HTM, Docker, Azure",
    ),
    (
        "Research on hyperparameters for ultrasonic signal classification",
        "academic",
        "Material classification combining signal processing and machine learning for real-time data "
        "acquisition.",
        "Python, Scikit-learn, NumPy, ultrasonic sensors",
    ),
    (
        "Reliability test and improvement of a sensor system for object detection",
        "academic",
        "Improved the accuracy and robustness of ultrasonic sensor-based object detection through "
        "machine-learning-driven adaptability to the environment.",
        "machine learning, ultrasonic sensors",
    ),
    (
        "Arecanut classifier and segregator",
        "academic",
        "Raspberry Pi based automated sorting system using colour-feature extraction and a motorized "
        "conveyor belt. Funded by KSCST and selected for the state-level KSCST seminar.",
        "Python, OpenCV, Raspberry Pi, image processing",
    ),
    (
        "ECU log failure prediction at Mercedes-Benz",
        "work",
        "AI system that reads ECU programming logs and predicts the cause of failure, reaching around "
        "90% prediction accuracy. Served through a Flask API with GUIs for live predictions and "
        "automatic ticket updates.",
        "Python, Pandas, Scikit-learn, Flask, Linux",
    ),
    (
        "Rexroth Control Unit (RCU) microservices at Bosch Rexroth",
        "work",
        "Microservices for a Linux IoT gateway: configuration feedback loop, remote force "
        "reconfiguration, status flush on ignition-off and remote Wi-Fi configuration, all over MQTT "
        "to the BODAS Connect cloud platform. Test-driven, with CI/CD.",
        "Go, Python, MQTT, Docker, Jenkins, Snap, TDD",
    ),
    (
        "Automated ARM build toolchain at Bosch Rexroth",
        "work",
        "Cross-compilation and packaging pipeline so Python applications built on Windows run on "
        "ARM-based embedded devices.",
        "Docker, Jenkins, Python, cross-compilation, Snap",
    ),
    (
        "Lateral flow test reader app at University Medical Center Mainz",
        "work",
        "Mobile app that turns a lateral flow test result into a reproducible 0-10 score from the "
        "test and control line intensities (T/C ratio), using native image processing.",
        "Flutter, Dart, C++, OpenCV, Dart FFI, Android",
    ),
    (
        "Clinical study data portal at University Medical Center Mainz",
        "work",
        "Self-service FastAPI website that lets researchers search, preview and export clinical study "
        "data from a normalized SQL database, using a Python ETL engine that pivots rows into tables.",
        "Python, FastAPI, SQL, ETL",
    ),
]


def seed(db_path: pathlib.Path = DB_PATH) -> int:
    """(Re)create the projects table. Returns the number of rows inserted."""
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS projects")
        conn.execute(
            """
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                technologies TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO projects (name, kind, description, technologies) VALUES (?, ?, ?, ?)",
            PROJECTS,
        )
        conn.commit()
    finally:
        conn.close()
    return len(PROJECTS)


if __name__ == "__main__":
    count = seed()
    print(f"Seeded {count} projects into {DB_PATH}")
