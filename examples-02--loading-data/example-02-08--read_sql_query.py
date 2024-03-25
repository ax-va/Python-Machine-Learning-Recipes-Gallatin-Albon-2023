import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# # # 1. Create SQLite database

# list of dictionaries
test_scores = [
    {
        'first_name': 'Jason',
        'last_name': 'Miller',
        'age': 42,
        'preTestScore': 4,
        'postTestScore': 25,
    },
    {
        'first_name': 'Molly',
        'last_name': 'Jacobson',
        'age': 52,
        'preTestScore': 24,
        'postTestScore': 94,
    },
]

# Create an engine for working with the database
engine = create_engine('sqlite:///../data/sqlite.db', echo=True)

# # # Use object-relational mapping (ORM)

# Use Base to define various tables
Base = declarative_base()


class TestScore(Base):
    """
    ORM: This class will correspond to the table, the instances
    of this class will correspond to the rows of the table,
    and the class attributes correspond to the columns of the table.
    """
    __tablename__ = 'TestScore'
    id = Column(Integer, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    age = Column(Integer)
    preTestScore = Column(Integer)
    postTestScore = Column(Integer)

    def __repr__(self):
        return f"<TestScore(first_name='{self.first_name}', last_name='{self.last_name}', age={self.age}, preTestScore={self.preTestScore}, postTestScore={self.postTestScore})>"


Base.metadata.create_all(engine)

jason_miller = TestScore(**test_scores[0])
molly_jacobson = TestScore(**test_scores[1])

# Create a session to interact with the database.
# Use the context manager.
Session = sessionmaker(bind=engine)

with Session() as session:
    # Add all the entries
    rows = [TestScore(**entry) for entry in test_scores]
    session.add_all(rows)
    # Alternatively:
    session.add(jason_miller)
    session.add(molly_jacobson)
    session.commit()


# # # 2. Read SQL query in pd.DataFrame

engine = create_engine('sqlite:///../data/sqlite.db')
dataframe = pd.read_sql_query('SELECT * FROM TestScore', engine)
dataframe.head()
#    id first_name last_name  age  preTestScore  postTestScore
# 0   1      Jason    Miller   42             4             25
# 1   2      Molly  Jacobson   52            24             94
# 2   3      Jason    Miller   42             4             25
# 3   4      Molly  Jacobson   52            24             94
