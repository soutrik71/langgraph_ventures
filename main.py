from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers"""
    return a / b


tools = [add, subtract, multiply, divide]


def main():
    print(add.invoke({"a": 1, "b": 2}))
    print(subtract.invoke({"a": 1, "b": 2}))
    print(multiply.invoke({"a": 1, "b": 2}))
    print(divide.invoke({"a": 1, "b": 2}))


if __name__ == "__main__":
    main()
