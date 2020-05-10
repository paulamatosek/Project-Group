# cli - command line interface aplikacji
# gui - graphical user interface
from ConnectSQL import ConnectionConfig
from Main import GroupProject


class CLI:
    def __init__(self):
        self.connection = ConnectionConfig().connection()
        d = GroupProject()
        d.dataPreprocessing()
        self.X_train, self.X_test, self.y_train, self.y_test = d.splitDatasetIntoTrainAndTest()

    def insertData(self,X,y):
        self.c =self.connection.cursor()
        index = 0
        while index<len(X['age']):
            self.c.execute("insert into task values(default, %s, %s, %s, %s,%s,%s, %s, %s, %s,%s,%s, %s, %s,%s)",
                       (X['age'][index], X['workclass'][index],X['fnlwgt'][index],X['education_num'][index],X['marital_status'][index],X['occupation'][index],X['relationship'][index],
                        X['race'][index],X['sex'][index],X['capital_gain'][index],X['capital_loss'][index],X['hours_per_week'][index],X['native_country'][index],y))
            index+=1
        decision = input("confirm added (Y/N)").upper()
        if (decision == 'Y'):
            self.conn.commit()  # zatwierdź i wprowadź do bazy danych
        else:
            self.conn.rollback()  # odrzuć dane i nie wprowadzaj do bazy danych
            print("nothing added")


    def menu(self):
        while(True):
            decision = input("(1) - train \n(2) - test \n(3) - results \n(4) - saveToData \n(Q) - return").upper()
            if (decision == "1"):
                pass
            elif (decision == "2"):
                pass
            elif (decision == "3"):
                pass
            elif (decision == "4"):
                self.insertData(self.X_train, self.y_train)

            elif (decision == "Q"):
                break
            else:
                print("wrong choice")


cli = CLI()

cli.menu()

