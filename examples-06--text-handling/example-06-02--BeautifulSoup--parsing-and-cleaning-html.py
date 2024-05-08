"""
Extract just the text from text data with HTML elements.
"""
from bs4 import BeautifulSoup  # lxml needed

HTML = (
    "<div class='full_name'>"
        "<span style='font-weight:bold'>"
            "  Masego"
        "</span>"
        " Azra"
    "</div>"
)

soup = BeautifulSoup(HTML, "lxml")
# <html><body><div class="full_name"><span style="font-weight:bold">    Masego</span> Azra</div></body></html>

# Find the div with the class "full_name", show text
soup.find(name="div", attrs={"class": "full_name"}).text
# '  Masego Azra'
