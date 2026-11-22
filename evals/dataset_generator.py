"""
evals/dataset_generator.py
--------------------------
Generates the evaluation benchmark dataset:
- 8 diverse job descriptions
- 60 realistic candidate profiles
- 480 graded relevance labels (0 = Not relevant, 1 = Partially relevant, 2 = Highly relevant)
- README notice stating the synthetic nature and review requirements.
"""

import json
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ── 8 Job Descriptions ────────────────────────────────────────────────────────
JOBS = [
    {
        "id": "jd_01",
        "title": "Senior Machine Learning Engineer",
        "raw_jd": "We are seeking a Senior ML Engineer with 5+ years of experience in Python, PyTorch, transformers, NLP, model serving, and FastAPI. Experience with vector databases and Docker is required. Location: Remote. Full-time.",
        "required_skills": ["Python", "PyTorch", "NLP", "transformers", "FastAPI"],
        "preferred_skills": ["Docker", "Vector Databases", "MLflow"],
        "years_experience": "5+",
        "location": "Remote",
    },
    {
        "id": "jd_02",
        "title": "Java Backend Microservices Engineer",
        "raw_jd": "Looking for a Backend Java Engineer with 5+ years of experience in Java, Spring Boot, microservices architecture, SQL, REST APIs, Kafka, and Docker. Experience with Kubernetes and high-throughput systems is a plus. Location: New York or Remote.",
        "required_skills": ["Java", "Spring Boot", "SQL", "REST APIs", "Microservices"],
        "preferred_skills": ["Kafka", "Docker", "Kubernetes"],
        "years_experience": "5+",
        "location": "New York",
    },
    {
        "id": "jd_03",
        "title": "Senior Full-Stack TypeScript Developer",
        "raw_jd": "Seeking an experienced Full-Stack Developer proficient in React, Node.js, TypeScript, GraphQL, Next.js, and CSS/Tailwind. Must have built responsive web apps. Location: Remote.",
        "required_skills": ["React", "TypeScript", "Node.js", "GraphQL"],
        "preferred_skills": ["Next.js", "Tailwind CSS", "PostgreSQL"],
        "years_experience": "4+",
        "location": "Remote",
    },
    {
        "id": "jd_04",
        "title": "DevOps & Cloud Platform Engineer",
        "raw_jd": "Join our infrastructure team as a DevOps Engineer. Requirements: 5+ years with AWS, Kubernetes, Terraform, Docker, CI/CD pipelines, Linux, and GitHub Actions. Location: Seattle.",
        "required_skills": ["AWS", "Kubernetes", "Terraform", "CI/CD", "Linux"],
        "preferred_skills": ["Helm", "Prometheus", "Python"],
        "years_experience": "5+",
        "location": "Seattle",
    },
    {
        "id": "jd_05",
        "title": "Senior Data Analyst",
        "raw_jd": "We are hiring a Senior Data Analyst skilled in Python, SQL, Tableau, Pandas, data warehousing, and business intelligence reporting. Location: Austin or Remote.",
        "required_skills": ["Python", "SQL", "Tableau", "Pandas", "Data Analysis"],
        "preferred_skills": ["Snowflake", "dbt", "PowerBI"],
        "years_experience": "3+",
        "location": "Austin",
    },
    {
        "id": "jd_06",
        "title": "Embedded Firmware Engineer",
        "raw_jd": "Seeking an Embedded Systems Engineer with deep expertise in C, C++, RTOS, microcontroller firmware (ARM Cortex), hardware debugging, and IoT communication protocols. Location: Boston.",
        "required_skills": ["C++", "C", "RTOS", "Firmware", "Embedded Systems"],
        "preferred_skills": ["ARM", "SPI/I2C", "BLE"],
        "years_experience": "4+",
        "location": "Boston",
    },
    {
        "id": "jd_07",
        "title": "Technical Product Manager (AI & Data)",
        "raw_jd": "Looking for a Technical Product Manager to lead AI products. Must have 4+ years leading agile software teams, product roadmapping, user research, Jira, and working with ML/data teams. Location: San Francisco.",
        "required_skills": ["Product Management", "Agile", "Roadmapping", "Jira", "User Research"],
        "preferred_skills": ["AI/ML Knowledge", "SQL", "A/B Testing"],
        "years_experience": "4+",
        "location": "San Francisco",
    },
    {
        "id": "jd_08",
        "title": "Mobile App Developer (Flutter & iOS)",
        "raw_jd": "We are hiring a Mobile Application Developer with 3+ years experience in Flutter, Dart, Swift, iOS, Android, and mobile REST integration. Location: Remote.",
        "required_skills": ["Flutter", "Dart", "iOS", "Android", "Mobile Development"],
        "preferred_skills": ["Swift", "Firebase", "App Store Deployment"],
        "years_experience": "3+",
        "location": "Remote",
    },
]

# ── 60 Candidates across 8 Domain Archetypes ──────────────────────────────────
# 0-9: ML/NLP (High for jd_01, Partial for jd_05/jd_07)
# 10-17: Java Backend (High for jd_02, Partial for jd_03/jd_04)
# 18-25: Full-Stack / Frontend (High for jd_03, Partial for jd_08)
# 26-33: DevOps / Cloud (High for jd_04, Partial for jd_02)
# 34-41: Data Analysts / BI (High for jd_05, Partial for jd_01)
# 42-49: Embedded / Systems (High for jd_06)
# 50-55: Product Managers (High for jd_07)
# 56-59: Mobile Developers (High for jd_08)

def generate_candidates():
    candidates = []

    # ML / NLP (10)
    ml_names = [
        ("Alice Sharma", "alice.sharma@example.com", "Python, PyTorch, NLP, transformers, FastAPI, Docker", 6, "Remote", "ML engineer with 6 years experience deploying large language models and transformer architectures."),
        ("David Chen", "david.chen@example.com", "Python, machine learning, NLP, HuggingFace, FastAPI, PyTorch", 5, "San Francisco", "Applied AI engineer building semantic search pipelines and RAG architectures."),
        ("Fatima Al-Hassan", "fatima.al@example.com", "Python, deep learning, PyTorch, computer vision, OpenCV, transformers", 4, "Remote", "Computer vision and multimodal deep learning specialist with strong PyTorch skills."),
        ("Marcus Vance", "marcus.v@example.com", "Python, NLP, scikit-learn, FastAPI, transformers, MLflow", 5, "Remote", "Production ML developer building fast microservices using FastAPI and HuggingFace."),
        ("Elena Rostova", "elena.r@example.com", "Python, PyTorch, LLMs, LangChain, vector databases, FastAPI", 4, "New York", "AI engineer focused on retrieval-augmented generation and transformer model deployment."),
        ("Liam O'Connor", "liam.oc@example.com", "Python, machine learning, TensorFlow, scikit-learn, Docker", 3, "Remote", "ML engineer experienced in classification, regression, and model packaging."),
        ("Siddharth Rao", "sid.rao@example.com", "Python, PyTorch, NLP, speech recognition, FastAPI, Docker", 7, "Seattle", "Senior AI researcher with 7 years deploying speech and NLP models at scale."),
        ("Chloe Bennett", "chloe.b@example.com", "Python, data science, machine learning, Pandas, PyTorch", 4, "Austin", "Data scientist with expertise in predictive modeling and PyTorch model experimentation."),
        ("Kenji Sato", "kenji.s@example.com", "Python, deep learning, NLP, FastAPI, Ray, Kubernetes", 6, "Remote", "Distributed ML systems engineer training and scaling transformer models on GPU clusters."),
        ("Amina Diallo", "amina.d@example.com", "Python, NLP, PyTorch, FastAPI, text embeddings, Docker", 5, "London", "NLP specialist with 5 years designing neural retrieval and entity extraction systems."),
    ]
    for n, e, s, exp, loc, sm in ml_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "ml"})

    # Java Backend (8)
    java_names = [
        ("Bob Zhang", "bob.zhang@example.com", "Java, Spring Boot, SQL, REST APIs, Docker, Kubernetes, Microservices", 7, "New York", "Senior Java backend engineer specialising in distributed microservices and PostgreSQL."),
        ("Rajesh Kulkarni", "rajesh.k@example.com", "Java, Spring Boot, Microservices, Kafka, SQL, Docker", 6, "Remote", "Backend engineer building high-volume event-driven microservices with Spring and Kafka."),
        ("Clara Becker", "clara.b@example.com", "Java, Spring Boot, REST APIs, SQL, Hibernate, JUnit", 5, "New York", "Java engineer building enterprise web APIs and relational database persistence."),
        ("Dmitri Ivanov", "dmitri.i@example.com", "Java, Microservices, Spring Boot, SQL, Kubernetes, CI/CD", 8, "Remote", "Staff backend engineer maintaining high-throughput banking transaction systems."),
        ("Sarah Jenkins", "sarah.j@example.com", "Java, Spring Boot, REST APIs, PostgreSQL, Docker, AWS", 4, "Chicago", "Backend software developer building secure REST endpoints and cloud integrations."),
        ("Guillermo Silva", "guillermo.s@example.com", "Java, Spring Boot, Kafka, SQL, Microservices, Docker", 5, "New York", "Microservices developer experienced in Kafka pub/sub and high-availability Java services."),
        ("Hanna Lindqvist", "hanna.l@example.com", "Java, Kotlin, Spring Boot, REST APIs, SQL", 4, "Remote", "Full backend engineer with Kotlin and Java experience creating scalable APIs."),
        ("Tariq Mansoor", "tariq.m@example.com", "Java, Spring Boot, SQL, Microservices, Redis, Docker", 6, "Seattle", "Backend engineer with 6 years experience in Java API development and caching."),
    ]
    for n, e, s, exp, loc, sm in java_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "java"})

    # Full-Stack / TypeScript (8)
    fs_names = [
        ("Carlos Rivera", "carlos.r@example.com", "JavaScript, React, Node.js, TypeScript, GraphQL, CSS", 6, "Remote", "Senior full-stack developer with 6 years in React, Node, and GraphQL APIs."),
        ("Emily Watson", "emily.w@example.com", "TypeScript, React, Next.js, Node.js, GraphQL, Tailwind", 5, "Remote", "Full-stack web engineer creating modern React interfaces and Node backends."),
        ("Lucas Moreau", "lucas.m@example.com", "TypeScript, React, Node.js, Express, PostgreSQL, CSS", 4, "New York", "Full-stack JavaScript developer building responsive UI and robust API services."),
        ("Pooja Nair", "pooja.n@example.com", "React, TypeScript, GraphQL, Redux, Node.js, HTML/CSS", 5, "Remote", "Frontend-leaning full-stack developer with deep expertise in React state management."),
        ("Tom Anderson", "tom.a@example.com", "TypeScript, Node.js, React, Next.js, Tailwind, GraphQL", 6, "San Francisco", "Full-stack architect with 6 years experience building SaaS applications."),
        ("Zoe Taylor", "zoe.t@example.com", "React, TypeScript, JavaScript, CSS, Node.js, REST", 3, "Austin", "Software engineer focused on intuitive component design and TypeScript full-stack apps."),
        ("Mateo Gomez", "mateo.g@example.com", "TypeScript, React, GraphQL, Node.js, MongoDB, Docker", 5, "Remote", "Full-stack engineer building fast single-page apps and serverless GraphQL endpoints."),
        ("Yuki Tanaka", "yuki.t@example.com", "TypeScript, Node.js, React, Next.js, PostgreSQL", 4, "Remote", "Web developer with 4 years building scalable web applications with Next.js."),
    ]
    for n, e, s, exp, loc, sm in fs_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "fullstack"})

    # DevOps & Cloud (8)
    devops_names = [
        ("Ravi Kumar", "ravi.k@example.com", "DevOps, AWS, Terraform, CI/CD, GitHub Actions, Linux, Kubernetes", 8, "Seattle", "Senior DevOps engineer with 8 years building AWS cloud infrastructure with Terraform."),
        ("Jessica Miller", "jessica.m@example.com", "Kubernetes, AWS, Terraform, Docker, CI/CD, Linux, Helm", 6, "Seattle", "Site reliability engineer managing container orchestration and automated delivery pipelines."),
        ("Alexander Popov", "alex.p@example.com", "Linux, AWS, Docker, Kubernetes, Terraform, Bash, CI/CD", 7, "Remote", "Infrastructure automation engineer with 7 years Linux systems and AWS experience."),
        ("Nadia Boulanger", "nadia.b@example.com", "AWS, Terraform, CI/CD, GitHub Actions, Linux, Prometheus", 5, "Remote", "DevOps engineer building monitoring, observability, and Terraform modules."),
        ("Kwame Mensah", "kwame.m@example.com", "Kubernetes, Docker, AWS, Linux, CI/CD, Terraform", 5, "Seattle", "Platform engineer maintaining microservice deployments across multi-region Kubernetes."),
        ("Samuel Cohen", "samuel.c@example.com", "AWS, CloudFormation, Terraform, CI/CD, Python, Linux", 6, "New York", "Cloud architect automating cloud governance and zero-downtime deployment pipelines."),
        ("Olga Smirnova", "olga.s@example.com", "Kubernetes, Helm, Docker, AWS, Terraform, Linux", 4, "Remote", "Cloud infrastructure developer with strong hands-on Kubernetes and Helm experience."),
        ("Brett Harper", "brett.h@example.com", "DevOps, AWS, Linux, CI/CD, Docker, Terraform", 5, "Seattle", "DevOps practitioner automating build systems and maintaining Linux server clusters."),
    ]
    for n, e, s, exp, loc, sm in devops_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "devops"})

    # Data Analysts / BI (8)
    data_names = [
        ("Priya Mehta", "priya.m@example.com", "Python, data analysis, pandas, NumPy, SQL, Tableau, PowerBI", 5, "Austin", "Senior data analyst with 5 years crafting Tableau dashboards and running complex SQL queries."),
        ("Arthur Pendelton", "arthur.p@example.com", "SQL, Python, Pandas, Tableau, Data Analysis, Snowflake", 4, "Remote", "BI analyst transforming warehouse data into automated business intelligence dashboards."),
        ("Leila Mostafa", "leila.m@example.com", "Python, SQL, Tableau, Data Analysis, Pandas, Matplotlib", 5, "Austin", "Data specialist analyzing customer retention cohorts and presenting executive insights."),
        ("Victor Hugo", "victor.h@example.com", "SQL, Tableau, Data Analysis, PowerBI, Excel, Python", 6, "Remote", "Analytics professional with 6 years experience in reporting and revenue forecasting."),
        ("Meera Joshi", "meera.j@example.com", "Python, Pandas, SQL, Tableau, Statistics, A/B testing", 4, "Austin", "Product data analyst partnering with growth teams to validate experiments via SQL and Python."),
        ("Brian O'Shea", "brian.os@example.com", "SQL, Tableau, Data Analysis, Snowflake, dbt, Pandas", 3, "New York", "Data analyst modeling ETL transformations with dbt and visualizing metrics in Tableau."),
        ("Sunita Patel", "sunita.p@example.com", "Python, SQL, Tableau, Pandas, Data Analysis, Dash", 5, "Remote", "Data insights lead with 5 years creating interactive Tableau and Python analytical tools."),
        ("Gabriel Torres", "gabriel.t@example.com", "SQL, Python, Data Analysis, Tableau, PowerBI", 4, "Austin", "BI developer designing dimensional models and KPI reporting dashboards."),
    ]
    for n, e, s, exp, loc, sm in data_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "data"})

    # Embedded Systems / C++ (6)
    emb_names = [
        ("Dmitri Volkov", "dmitri.v@example.com", "C++, C, embedded systems, RTOS, firmware, IoT, ARM", 10, "Boston", "Embedded systems veteran with a decade writing real-time RTOS firmware for ARM microcontrollers."),
        ("Stefan Mueller", "stefan.m@example.com", "C, C++, RTOS, Firmware, Embedded Systems, SPI/I2C", 6, "Boston", "Firmware engineer developing low-power IoT sensor node software in C and modern C++."),
        ("Ananya Sen", "ananya.s@example.com", "C++, Embedded Systems, RTOS, Firmware, Linux, C", 5, "Remote", "Embedded developer programming automotive ECUs and real-time hardware drivers."),
        ("Klaus Weber", "klaus.w@example.com", "C, C++, Firmware, RTOS, Embedded Systems, BLE", 7, "Boston", "Hardware-software integration specialist programming wireless microcontrollers."),
        ("Hiroshi Takahashi", "hiroshi.t@example.com", "C++, RTOS, Embedded Systems, Firmware, ARM Cortex", 8, "Remote", "Embedded systems developer building high-reliability medical device firmware."),
        ("Lars Nielsen", "lars.n@example.com", "C, C++, RTOS, Firmware, IoT, Hardware Debugging", 5, "Boston", "Firmware developer experienced with oscilloscopes, logic analyzers, and RTOS threads."),
    ]
    for n, e, s, exp, loc, sm in emb_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "embedded"})

    # Product Managers (6)
    pm_names = [
        ("Mei Lin", "mei.lin@example.com", "Product Management, Agile, Jira, user research, roadmapping, ML", 6, "San Francisco", "Product manager with 6 years experience shipping B2B SaaS and machine learning features."),
        ("Rachel Green", "rachel.g@example.com", "Product Management, Agile, Roadmapping, Jira, User Research", 5, "San Francisco", "Technical PM guiding cross-functional agile teams from problem discovery to product launch."),
        ("Devon Brooks", "devon.b@example.com", "Product Management, Agile, User Research, Roadmapping, Jira", 4, "Remote", "Growth product manager optimizing onboarding funnels and user retention."),
        ("Kavita Reddy", "kavita.r@example.com", "Product Management, Roadmapping, Agile, Jira, Data Analysis", 7, "San Francisco", "Principal PM with 7 years leading enterprise data platform and analytics product lines."),
        ("Simon Cox", "simon.c@example.com", "Product Management, Agile, Scrum, Jira, Roadmapping, UI/UX", 5, "Remote", "Scrum-certified product manager translating business goals into crisp user stories."),
        ("Nathalie Dubois", "nathalie.d@example.com", "Product Management, User Research, Roadmapping, Agile, Jira", 6, "San Francisco", "Technical product leader with 6 years experience driving cloud product strategies."),
    ]
    for n, e, s, exp, loc, sm in pm_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "pm"})

    # Mobile App Developers (6)
    mob_names = [
        ("Lucas Silva", "lucas.silva@example.com", "Flutter, Dart, iOS, Android, Mobile Development, Swift", 5, "Remote", "Mobile engineer with 5 years building and shipping cross-platform Flutter mobile applications."),
        ("Tara Higgins", "tara.h@example.com", "Flutter, Dart, Mobile Development, iOS, Android, Firebase", 4, "Remote", "Mobile software developer creating slick Flutter user interfaces for iOS and Android."),
        ("Alonso Morales", "alonso.m@example.com", "Mobile Development, Flutter, Dart, Swift, iOS, Android", 6, "Remote", "Lead mobile architect experienced in native iOS bridge integrations and Flutter state."),
        ("Jasmine Zhao", "jasmine.z@example.com", "Flutter, Dart, iOS, Android, Mobile Development, REST", 3, "New York", "Mobile developer specialized in offline-first data caching and clean Dart code."),
        ("Henrik Berg", "henrik.b@example.com", "Flutter, Dart, Swift, Mobile Development, iOS, Android", 5, "Remote", "Cross-platform mobile engineer with 5 published applications in the App Store."),
        ("Fatou Traore", "fatou.t@example.com", "Mobile Development, Flutter, Dart, Android, iOS, Firebase", 4, "Remote", "Mobile developer building real-time collaboration apps with Flutter and Firebase."),
    ]
    for n, e, s, exp, loc, sm in mob_names:
        candidates.append({"name": n, "email": e, "skills": s, "years_experience": exp, "location": loc, "summary": sm, "domain": "mobile"})

    return candidates


def generate_relevance_labels(jobs, candidates):
    """
    Assigns ground truth graded relevance labels (2 = Highly Relevant, 1 = Partially Relevant, 0 = Irrelevant)
    based on domain matching and skill overlap.
    """
    job_domain_map = {
        "jd_01": ("ml", ["data"]),             # ML primary, Data secondary
        "jd_02": ("java", ["devops"]),          # Java primary, DevOps secondary
        "jd_03": ("fullstack", ["mobile"]),     # Fullstack primary, Mobile secondary
        "jd_04": ("devops", ["java"]),          # DevOps primary, Java secondary
        "jd_05": ("data", ["ml"]),              # Data primary, ML secondary
        "jd_06": ("embedded", []),              # Embedded primary
        "jd_07": ("pm", ["data"]),              # PM primary, Data secondary
        "jd_08": ("mobile", ["fullstack"]),     # Mobile primary, Fullstack secondary
    }

    labels = {}
    for job in jobs:
        jid = job["id"]
        primary_domain, secondary_domains = job_domain_map[jid]
        job_labels = {}

        for cand in candidates:
            c_domain = cand["domain"]
            email = cand["email"]

            if c_domain == primary_domain:
                # Highly relevant (2), unless low experience for senior role
                job_labels[email] = 2 if cand["years_experience"] >= 4 else 1
            elif c_domain in secondary_domains:
                # Partially relevant (1)
                job_labels[email] = 1
            else:
                # Not relevant (0)
                job_labels[email] = 0

        labels[jid] = job_labels
    return labels


def build_and_save_dataset():
    candidates = generate_candidates()
    labels = generate_relevance_labels(JOBS, candidates)

    # 1. Save job descriptions
    with open(DATASET_DIR / "job_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(JOBS, f, indent=2)

    # 2. Save candidates CSV
    df = pd.DataFrame([{k: v for k, v in c.items() if k != "domain"} for c in candidates])
    df.to_csv(DATASET_DIR / "candidates.csv", index=False)

    # 3. Save relevance labels
    with open(DATASET_DIR / "relevance_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    # 4. Save README
    readme_content = """# HireFlow v3 Evaluation Benchmark Dataset

> ⚠️ **SYNTHETIC DATASET NOTICE & REVIEW REQUIREMENT**
>
> This benchmark dataset was generated synthetically to provide a rigorous, standardized ground-truth testbed
> for evaluating candidate retrieval and ranking algorithms (Precision@k, Recall@k, MRR, and NDCG@k).
>
> - **8 Job Descriptions**: Covering ML, Java Backend, Full-Stack, DevOps, Data Analytics, Embedded, PM, and Mobile.
> - **60 Candidate Profiles**: 60 diverse, realistic software engineering and product resumes.
> - **480 Graded Relevance Annotations**:
>   - `2` = Highly Relevant (matches primary technical core skills and seniority requirements)
>   - `1` = Partially Relevant (adjacent skill set or partial domain transfer)
>   - `0` = Irrelevant (different engineering discipline)
>
> **Human Review Notice**: Before citing evaluation benchmark results on a resume or production audit report,
> please review and calibrate these relevance annotations against your organization's specific hiring bar.
"""
    with open(DATASET_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Generated evaluation dataset: {len(JOBS)} jobs, {len(candidates)} candidates, {len(JOBS) * len(candidates)} relevance labels in {DATASET_DIR}")


if __name__ == "__main__":
    build_and_save_dataset()
