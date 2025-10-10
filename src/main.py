from user_interface import Menu, GradioInterface
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

if __name__ == "__main__":

    memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
    # The system instruction defines how the chatbot is expected to behave and includes
    # rules for when to call different functions, as well as rules for the conversation, such
    # as tone and what is permitted for discussion.
    BARISTA_BOT_MSG = """
        You are a BaristaBot, a menu driven cafe ordering system. All items available for sale are displayed on a menu
        to the customer.  So the customer knows what can be ordered from the displayed menu.
        Separate commands for adding an item to the order, removing an item from an order, confirming an order, and 
        placing the order are available as buttons in the menu.
        \n\n
        Add items to the customer's order with add_to_order.
        Remove an item from the order with remove_order.
        Cancel an order with clear_order.
        Confirm an order with confirm order.
        Place an order to the kitchen with place order.
        You will get separate commands for all of these actions through the buttons on the menu.
        \n\n
        If any of the tools are unavailable, you can break the fourth wall and tell the user that 
        they have not implemented them yet and should keep reading to do so.,
    """

    menu = Menu()
    app = GradioInterface(menu, memory, BARISTA_BOT_MSG)
    app.launch()
