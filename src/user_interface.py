from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from collections.abc import Iterable
from random import randint
from typing import List, Optional
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import Any, TypedDict, Annotated
import gradio as gr
import pandas as pd
import logging
import re
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GOOGLE_API_KEY"] = "TBD"


class OrderState(TypedDict):
    """State representing the customer's order conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress order.
    order: list[str]

    # Flag indicating that the order is placed and completed.
    finished: bool
    user_email: str

    # remove index
    remove_index: int


class DrinkOrder(BaseModel):
    """
    Represents a drink order with various options and modifiers.
    """
    drink_type: str
    milk_option: str = "Whole"
    espresso_shots: str = "Double"
    caffeine: str = "Regular"
    hot_iced: str = "Hot"
    sweeteners: List[str] = []
    special_requests: Optional[str] = None
    dirty: bool = False
    sweetened: bool = False

    @field_validator('milk_option')
    def milk_option_check(cls, v: str, info: ValidationInfo) -> str:
        """
        Validates the milk option, excluding "Soy" as it's out of stock.
        """
        if v.lower() == "soy":
            raise ValueError("Soy milk is currently unavailable.")
        return v


class Menu:
    """
    Defines the drink menu with categories and items.
    """

    def __init__(self):
        self.menu = {
            "Coffee Drinks": ["Espresso", "Americano", "Cold Brew"],
            "Coffee Drinks with Milk": ["Latte", "Cappuccino", "Cortado", "Macchiato", "Mocha",
                                        "Flat White"],
            "Tea Drinks": ["English Breakfast Tea", "Green Tea", "Earl Grey"],
            "Tea Drinks with Milk": ["Chai Latte", "Matcha Latte", "London Fog"],
            "Other Drinks": ["Steamer", "Hot Chocolate"]
        }
        self.milk_options = ["None", "Whole", "2%", "Oat", "Almond", "2% Lactose Free"]
        self.espresso_shots = ["None", "Single", "Double", "Triple", "Quadruple"]
        self.caffeine_options = ["Decaf", "Regular"]
        self.temperature_options = ["Hot", "Iced"]
        self.sweetener_options = ["None", "vanilla sweetener", "hazelnut sweetener", "caramel sauce",
                                  "chocolate sauce", "sugar free vanilla sweetener"]

    def get_drinks_list(self) -> List[str]:
        """
        Returns a list of all available drinks.
        """
        return [drink for category in self.menu.values() for drink in category]

    def create_drink_description(self, order: DrinkOrder) -> str:
        """
        Generates a description of the drink order.
        """
        milk_description = f" with {order.milk_option} milk"
        if order.milk_option == "None":
            milk_description = ""
        espresso_description = f", {order.espresso_shots} shot(s) of {order.caffeine} caffeine."
        if order.espresso_shots == "None":
            espresso_description = "."
        description = f"A {order.hot_iced} {order.drink_type}{milk_description}" \
                      f"{espresso_description}"

        if order.sweetened:
            description += " Sweetened with regular sugar."
        if order.sweeteners:
            description += f" With sweeteners: {', '.join(order.sweeteners)}."
        if order.dirty:
            description += " Dirty (with an extra shot of espresso)."
        if order.special_requests:
            description += f" Special requests: {order.special_requests}."

        return description


# These functions have no body; LangGraph does not allow @tools to update
# the conversation state, so you will implement a separate node to handle
# state updates. Using @tools is still very convenient for defining the tool
# schema, so empty functions have been defined that will be bound to the LLM
# but their implementation is deferred to the order_node.

@tool
def get_menu() -> str:
    """Provide the latest up-to-date menu."""
    # Note that this is just hard-coded text, but you could connect this to a live stock
    # database, or you could use Gemini's multi-modal capabilities and take live photos of
    # your cafe's chalk menu or the products on the counter and assmble them into an input.
    return """
      MENU:
      Coffee Drinks:
      Espresso
      Americano
      Cold Brew

      Coffee Drinks with Milk:
      Latte
      Cappuccino
      Cortado
      Macchiato
      Mocha
      Flat White

      Tea Drinks:
      English Breakfast Tea
      Green Tea
      Earl Grey

      Tea Drinks with Milk:
      Chai Latte
      Matcha Latte
      London Fog

      Other Drinks:
      Steamer
      Hot Chocolate

      Modifiers:
      Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
      Espresso shots: Single, Double, Triple, Quadruple; default: Double
      Caffeine: Decaf, Regular; default: Regular
      Hot-Iced: Hot, Iced; Default: Hot
      Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
      Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

      "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
      "Regular milk" is the same as 'whole milk'.
      "Sweetened" means add some regular sugar, not a sweetener.

      Soy milk has run out of stock today, so soy is not available.
    """


@tool
def add_to_order(drink: str, modifiers: Iterable[str]) -> str:
    """Adds the specified drink to the customer's order, including any modifiers.

    Returns:
      The updated order in progress.
    """


@tool
def remove_from_order(remove_index: int) -> str:
    """Removes the specified drink from the customer's order using the index.

    Returns:
      The updated order in progress.
    """


@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct.

    Returns:
      The user's free-text response.
    """


@tool
def get_order() -> str:
    """Returns the users order so far. One item per line."""


@tool
def cancel_order():
    """Removes all items from the user's order."""


@tool
def place_order() -> int:
    """Sends the order to the barista for fulfillment.

    Returns:
      The estimated number of minutes until the order is ready.
    """


class GradioInterface:
    """
    Defines the Gradio interface for ordering drinks.
    """
    ORDER_START: str = "*** Start ***"
    ORDER_END: str = "*** Complete ***"
    ORDER_ITEM: str = "Order Item"
    SELECTED: str = "Selected"
    ADD_TO_ORDER: str = "Please add the following to my order"
    REMOVE_FROM_ORDER: str = "Please remove the following from my order"
    CANCEL_ORDER: str = "Please cancel my order"
    CONFIRM_ORDER: str = "Please confirm my order"
    PLACE_CONFIRMED_ORDER: str = "Please place my order"

    def __init__(self, menu: Menu, memory: SqliteSaver, system_prompt: str, share=False):
        self.menu = menu
        self.df_data = pd.DataFrame({self.ORDER_ITEM: [self.ORDER_START], self.SELECTED: [False]})
        self.system_prompt = system_prompt
        self.order_tools = [add_to_order, remove_from_order, confirm_order, get_order, cancel_order, place_order]
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.share = share
        self.email = None

        # Define a new graph
        self.workflow = StateGraph(OrderState)
        self.llm_prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("placeholder", "{messages}")]
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                          temperature=0,
                                          max_tokens=8196,
                                          timeout=None,
                                          max_retries=2,
                                          max_output_tokens=8196)
        self.llm_with_order_tools = self.llm.bind_tools(self.order_tools)
        self.llm_gen = (self.llm_prompt | self.llm_with_order_tools)
        # Nodes
        self.workflow.add_node("get_menu_msg", self.get_menu_msg)
        # Add nodes for the get_menu
        self.workflow.add_node(
            "get_menu_tool", self.create_tool_node_with_fallback([get_menu])
        )
        self.workflow.add_node("chatbot_with_tools", self.chatbot_with_tools)
        # Specify the edges between the nodes
        self.workflow.add_edge(START, "get_menu_msg")
        self.workflow.add_edge("get_menu_msg", "get_menu_tool")
        self.workflow.add_edge("get_menu_tool", "chatbot_with_tools")
        self.workflow.add_edge("chatbot_with_tools", END)
        self.memory = memory
        self.graph = self.workflow.compile(checkpointer=self.memory)
        self.demo = self.create_interface()

    # Add a node for the first tool call
    def get_menu_msg(self, state: OrderState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_menu",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }

    def create_tool_node_with_fallback(self, tools: list) -> RunnableWithFallbacks[Any, dict]:
        """
        Create a ToolNode with a fallback to handle errors and surface them to the agent.
        """
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )

    def handle_tool_error(self, state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def chatbot_with_tools(self, state: OrderState) -> OrderState:
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        order = state.get("order", [])
        finished = state.get("finished", False)
        messages = state['messages']
        remove_index = state.get("remove_index", -1)

        if state["messages"]:
            new_output = self.llm_gen.invoke(state)
            messages.append(new_output);
            for tool_call in new_output.tool_calls:
                if tool_call["name"] == "add_to_order":

                    # Each order item is just a string. This is where it assembled as "drink (modifiers, ...)".
                    modifiers = tool_call["args"]["modifiers"]
                    modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

                    order.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
                    response = "\n".join(order)
                elif tool_call["name"] == "remove_from_order":
                    if remove_index > -1:
                        order.pop(remove_index)
                    response = "\n".join(order)
                elif tool_call["name"] == "confirm_order":
                    finished = True;
                    response = finished
                elif tool_call["name"] == "get_order":
                    response = "\n".join(order) if order else "(no order)"
                elif tool_call["name"] == "cancel_order":
                    order.clear()
                    response = None
                elif tool_call["name"] == "place_order":
                    order_text = "\n".join(order)
                    order.clear()
                    finished = False;
                    response = order_text + "\n ETA in " + str(randint(1, 5)) + " minute(s)"  # ETA in minutes
                else:
                    raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

                # Record the tool results as tool messages.
                state["finished"] = finished
                state["order"] = order
                messages.append(
                    ToolMessage(
                        content=response,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                state["messages"] = messages
        else:
            messages.append = AIMessage(content=self.model_prompt)
            state["messages"] = messages
        return state

    def create_interface(self) -> gr.Blocks:
        """
        Creates the Gradio Blocks interface.
        """
        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm', text_size="sm")) as demo:
            gr.HTML('''
            <style>
            .custom-width {
              width: 300px !important;
            }
            </style>
            ''')
            with gr.Row():
                gr.Markdown("## Welcome to AI Barista Agent!")
            with gr.Row():
                with gr.Column(scale=1):
                    email = gr.Textbox(label="Email",
                                       placeholder="Please enter you email address",
                                       info="Enter your email and hit the Enter Key to activate the menu",
                                       lines=1)
                with gr.Column(scale=2):
                    gr.Markdown(
                        """
                        **Note:**
                        - "dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
                        - "Regular milk" is the same as 'whole milk'.
                        - "Sweetened" means add some regular sugar, not a sweetener.
                        - Soy milk has run out of stock today, so soy is not available.
                        """
                    )
            with gr.Row():
                with gr.Column():
                    llm_log = gr.HTML()
            with gr.Row():
                drink_type = gr.Dropdown(choices=self.menu.get_drinks_list(), label="Drink Type", scale=0,
                                         min_width=150)
                milk_option = gr.Radio(choices=self.menu.milk_options, value="None", label="Milk Option", scale=0,
                                       min_width=200)
                espresso_shots = gr.Radio(choices=self.menu.espresso_shots, value="None",
                                          label="Espresso Shots", scale=0, min_width=150)
                caffeine = gr.Radio(choices=self.menu.caffeine_options, value="Regular", label="Caffeine", scale=0,
                                    min_width=100)
                hot_iced = gr.Radio(choices=self.menu.temperature_options, value="Hot", label="Hot/Iced", scale=0,
                                    min_width=100)
                sweeteners = gr.CheckboxGroup(choices=self.menu.sweetener_options, label="Sweeteners", scale=1,
                                              min_width=150)
                special_requests = gr.Textbox(label="Special Requests", placeholder="e.g., extra hot, one pump",
                                              scale=0, min_width=150)
                dirty = gr.Checkbox(label="Dirty (add espresso shot)", scale=0, min_width=150)
                sweetened = gr.Checkbox(label="Sweetened (add regular sugar)", scale=0, min_width=150)
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    add_to_order_button = gr.Button("Add", interactive=False, size="sm")
                with gr.Column(scale=0, min_width=50):
                    del_from_order_button = gr.Button("Delete", interactive=False, size="sm")
                with gr.Column(scale=0, min_width=50):
                    complete_order_button = gr.Button("Confirm", interactive=False, size="sm")
                with gr.Column(scale=0, min_width=50):
                    cancel_order_button = gr.Button("Cancel", interactive=False, size="sm")
                with gr.Column(scale=0, min_width=50):
                    place_order_button = gr.Button("Place", interactive=False, size="sm")
                with gr.Column(scale=1, min_width=400):
                    current_order = gr.DataFrame(
                        headers=[self.ORDER_ITEM, self.SELECTED],
                        datatype=['str', 'bool'],
                        value=self.df_data,
                        label="Current Order",
                        interactive=False,
                        column_widths=['85%', '15%'],
                        show_search="search",
                        show_copy_button=True,
                        show_row_numbers=True,
                        show_fullscreen_button=True
                    )
            email.submit(self.check_valid_email,
                         inputs=[email],
                         outputs=[add_to_order_button,
                                  complete_order_button,
                                  del_from_order_button,
                                  cancel_order_button,
                                  place_order_button,
                                  llm_log])
            add_to_order_button.click(
                self.create_order_description,
                inputs=[drink_type, milk_option, espresso_shots, caffeine, hot_iced, sweeteners,
                        special_requests, dirty, sweetened, current_order],
                outputs=[current_order, complete_order_button, cancel_order_button, llm_log]
            )
            del_from_order_button.click(
                self.delete_order_item,
                inputs=[current_order],
                outputs=[current_order,
                         del_from_order_button,
                         complete_order_button,
                         cancel_order_button,
                         llm_log]
            )
            cancel_order_button.click(
                self.cancel_order,
                inputs=[current_order],
                outputs=[current_order,
                         del_from_order_button,
                         complete_order_button,
                         cancel_order_button,
                         llm_log]
            )
            complete_order_button.click(
                self.complete_order,
                inputs=[current_order],
                outputs=[current_order,
                         add_to_order_button,
                         del_from_order_button,
                         complete_order_button,
                         cancel_order_button,
                         place_order_button,
                         llm_log]
            )
            place_order_button.click(
                self.place_order,
                inputs=[current_order],
                outputs=[current_order,
                         add_to_order_button,
                         del_from_order_button,
                         complete_order_button,
                         cancel_order_button,
                         place_order_button,
                         llm_log,
                         email]
            )
            current_order.select(
                fn=self.on_select_cell,
                inputs=current_order,
                outputs=[current_order, del_from_order_button]
            )

        return demo

    def get_llm_log(self):
        current_state = self.graph.get_state(self.thread)
        order = current_state.values.get("order", [])
        finished = current_state.values.get("finished", False)
        user_email = current_state.values.get("user_email", "")
        llm_log = f"<table><tr><td colspan='2'><b>AI Barista Status</b></td></tr>"
        llm_log += f"<tr><td>Email</td><td><b>{user_email}</b></td></tr>"
        llm_log += f"<tr><tr><td>Finished</td><td><b>{str(finished)}</b></td></tr>"
        for i in range(0, len(order)):
            if i == 0:
                llm_log += f"<tr><tr><td>Order</td><td><b>{order[i]}</b></td></tr>"
            else:
                llm_log += f"<tr><tr><td></td><td><b>{order[i]}</b></td></tr>"
        llm_log += f"</table>"
        return llm_log

    def create_order_description(self,
                                 drink_type: str,
                                 milk_option: str,
                                 espresso_shots: str,
                                 caffeine: str,
                                 hot_iced: str,
                                 sweeteners: List[str],
                                 special_requests: str,
                                 dirty: bool,
                                 sweetened: bool,
                                 current_order):
        """
        Creates a drink order and generates its description.
        """
        try:
            order = DrinkOrder(
                drink_type=drink_type,
                milk_option=milk_option,
                espresso_shots=espresso_shots,
                caffeine=caffeine,
                hot_iced=hot_iced,
                sweeteners=sweeteners,
                special_requests=special_requests,
                dirty=dirty,
                sweetened=sweetened
            )
            my_order_description = self.menu.create_drink_description(order)
            user_msg = f"{self.ADD_TO_ORDER} {my_order_description}"
            current_state = self.graph.get_state(self.thread)
            order = current_state.values.get("order", [])
            finished = current_state.values.get("finished", False)
            remove_index = -1
            model_qry = {
                "messages": [("user", user_msg)],
                "order": order,
                "finished": finished,
                "user_email": self.email,
                "remove_index": remove_index
            }
            self.graph.invoke(model_qry, self.thread)
            updated_state = self.graph.get_state(self.thread)
            order = updated_state.values.get("order", [])
            my_order_description = order[len(order) - 1]
            new_row = pd.DataFrame({self.ORDER_ITEM: [my_order_description], self.SELECTED: [False]})
            updated_order = pd.concat([current_order, new_row], ignore_index=True)
            llm_log = self.get_llm_log()
            return (updated_order,
                    gr.Button("Confirm", interactive=True),
                    gr.Button("Cancel", interactive=True),
                    llm_log)

        except ValueError as e:
            logging.error(f"Order creation failed: {e}")
            return f"Error: {e}"
        except Exception as e:
            logging.exception("An unexpected error occurred.")
            return "An unexpected error occurred. Please check the logs."

    def delete_order_item(self, current_order: pd.DataFrame):
        """
        Deletes the selected rows from the DataFrame.
        Args:
        df (pd.DataFrame): The input DataFrame.
        Returns:
        pd.DataFrame: The updated DataFrame after deleting the rows.
        """
        try:
            num_rows: int = 0
            df = current_order
            for index, row in current_order.iterrows():
                if row['Selected'] and row[self.ORDER_ITEM] != self.ORDER_START:
                    my_order_description = row[self.ORDER_ITEM]
                    user_msg = f"{self.REMOVE_FROM_ORDER} {my_order_description}"
                    current_state = self.graph.get_state(self.thread)
                    order = current_state.values.get("order", [])
                    remove_index = order.index(my_order_description)
                    finished = current_state.values.get("finished", False)
                    model_qry = {
                        "messages": [("user", user_msg)],
                        "order": order,
                        "finished": finished,
                        "user_email": self.email,
                        "remove_index": remove_index
                    }
                    self.graph.invoke(model_qry, self.thread)
                    num_rows = num_rows + 1
                    df = current_order.drop(index)
                    llm_log = self.get_llm_log()
            is_interactive = True
            if len(df) < 2:
                is_interactive = False
            logging.info(f"Deleted rows {num_rows}. Updated DataFrame: {df}")
            return (df,
                    gr.Button("Delete", interactive=False),
                    gr.Button("Confirm", interactive=is_interactive),
                    gr.Button("Cancel", interactive=is_interactive),
                    llm_log)
        except Exception as e:
            logging.error(f"Error deleting rows: {e}")
            raise

    def cancel_order(self, current_order: pd.DataFrame):
        """
        Cancels the order.
        Args:
        df (pd.DataFrame): The input DataFrame.
        Returns:
        pd.DataFrame: The updated DataFrame after deleting the rows.
        """
        try:
            num_rows: int = 0
            remove_index = -1
            current_state = self.graph.get_state(self.thread)
            order = current_state.values.get("order", [])
            finished = current_state.values.get("finished", False)
            model_qry = {
                "messages": [("user", self.CANCEL_ORDER)],
                "order": order,
                "finished": finished,
                "user_email": self.email,
                "remove_index": remove_index
            }
            self.graph.invoke(model_qry, self.thread)
            df = current_order
            for index, row in current_order.iterrows():
                if row[self.ORDER_ITEM] != self.ORDER_START and row[self.ORDER_ITEM] != self.ORDER_END:
                    num_rows = num_rows + 1
                    df = df.drop(index)
            llm_log = self.get_llm_log()
            logging.info(f"Deleted rows {num_rows}. Updated DataFrame: {df}")
            return (df,
                    gr.Button("Delete", interactive=False),
                    gr.Button("Confirm", interactive=False),
                    gr.Button("Cancel", interactive=False),
                    llm_log)
        except Exception as e:
            logging.error(f"Error deleting rows: {e}")
            raise

    def complete_order(self, current_order):
        """
        Creates a drink order and generates its description.
        """
        try:
            remove_index = -1
            current_state = self.graph.get_state(self.thread)
            order = current_state.values.get("order", [])
            finished = current_state.values.get("finished", False)
            model_qry = {
                "messages": [("user", self.CONFIRM_ORDER)],
                "order": order,
                "finished": finished,
                "user_email": self.email,
                "remove_index": remove_index
            }
            self.graph.invoke(model_qry, self.thread)
            llm_log = self.get_llm_log()
            new_row = pd.DataFrame({self.ORDER_ITEM: [self.ORDER_END], self.SELECTED: [False]})
            updated_order = pd.concat([current_order, new_row], ignore_index=True)
            return (updated_order,
                    gr.Button("Add", interactive=False),
                    gr.Button("Delete", interactive=False),
                    gr.Button("Confirm", interactive=False),
                    gr.Button("Cancel", interactive=False),
                    gr.Button("Place", interactive=True),
                    llm_log)
        except ValueError as e:
            logging.error(f"Order creation failed: {e}")
            return f"Error: {e}"
        except Exception as e:
            logging.exception("An unexpected error occurred.")
            return "An unexpected error occurred. Please check the logs."

    def place_order(self, current_order):
        """
        Creates a drink order and generates its description.
        """
        try:
            remove_index = -1
            current_state = self.graph.get_state(self.thread)
            order = current_state.values.get("order", [])
            finished = current_state.values.get("finished", False)
            model_qry = {
                "messages": [("user", self.PLACE_CONFIRMED_ORDER)],
                "order": order,
                "finished": finished,
                "user_email": self.email,
                "remove_index": remove_index
            }
            response = self.graph.invoke(model_qry, self.thread)
            my_message = response["messages"][-1].content
            llm_log = f"<h1>Thank you for your order!</h1><h2>{my_message}</h2>"
            my_email = ""
            updated_order = pd.DataFrame({self.ORDER_ITEM: [self.ORDER_START], self.SELECTED: [False]})
            return (updated_order,
                    gr.Button("Add", interactive=False),
                    gr.Button("Delete", interactive=False),
                    gr.Button("Confirm", interactive=False),
                    gr.Button("Cancel", interactive=False),
                    gr.Button("Place", interactive=False),
                    llm_log,
                    my_email)
        except ValueError as e:
            logging.error(f"Order creation failed: {e}")
            return f"Error: {e}"
        except Exception as e:
            logging.exception("An unexpected error occurred.")
            return "An unexpected error occurred. Please check the logs."

    def on_select_cell(self, current_order: pd.DataFrame, evt: gr.SelectData):
        """
        This function is called when a cell in the Dataframe is selected.
        It receives a gr.SelectData object containing details about the selection.
        """
        df = current_order
        is_interactive, df = self.update_df_data_selection(df, evt)
        return df, gr.Button("Delete", interactive=is_interactive)

    def check_valid_email(self, email):
        is_valid = bool(re.search(r"^[\w\.\+\-]+\@[\w]+\.[a-z]{2,3}$", email))
        llm_log = ""
        if is_valid:
            self.email = email
        return (gr.Button("Add", interactive=is_valid),
                gr.Button("Complete", interactive=False),
                gr.Button("Delete", interactive=False),
                gr.Button("cancel", interactive=False),
                gr.Button("place", interactive=False),
                llm_log)

    def update_df_data_selection(self, current_order: pd.DataFrame, evt: gr.SelectData):
        if not self.is_order_complete(current_order):
            for index, row in current_order.iterrows():
                if (index == evt.index[0] and
                        row[self.ORDER_ITEM] != self.ORDER_START and
                        row[self.ORDER_ITEM] != self.ORDER_END):
                    update_selected: bool = not row['Selected']
                    current_order.loc[index, self.SELECTED] = not row['Selected']
                    return update_selected, current_order
        return False, current_order

    def is_order_complete(self, current_order: pd.DataFrame) -> bool:
        order_complete: bool = False
        for index, row in current_order.iterrows():
            if row[self.ORDER_ITEM] == self.ORDER_END:
                order_complete = True
        return order_complete

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
