from flask import (
    render_template,
    request,
    redirect,
    url_for,
    Blueprint,
    session,
)

main = Blueprint('display', __name__)


@main.route("/", methods=['POST', 'GET'])
def display():
    if request.method=='GET':
        content = "666"
        return render_template("display.html", content=content)
    else:
        thiscontent=request.form
        return render_template("display.html",content=thiscontent)

