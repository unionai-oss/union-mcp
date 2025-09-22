from union_mcp.v1.server import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio")
