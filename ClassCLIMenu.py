# cli - command line interface aplikacji
# gui - graphical user interface
from Connect import ConnectionConfig


class CLI:
    def __init__(self):
        self.connection = ConnectionConfig().connection()

    def menu(self):
        while(True):
            decision = input("(1) - train \n(2) - test \n(3) - results \n(Q) - return").upper()
            if (decision == "1"):
                pass
            elif (decision == "2"):
                pass
            elif (decision == "3"):
                pass
            elif (decision == "Q"):
                break
            else:
                print("wrong choice")


cli = CLI()

cli.menu()

