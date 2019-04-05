from flask import (
    Flask,
    render_template,
    request,
)
from flask import send_file, send_from_directory
import os
import datetime
import time
from routes.display import main as display_routes
from routes.generate import Generate

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN']=True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True
app.secret_key = "EST"
app.register_blueprint(display_routes, url_prefix='/display')


@app.route("/xlq")
def display():
    # content="666"
    return render_template("display.html")


@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']

    print('username:' + username)
    print('password:' + password)
    return '注册成功'


@app.route('/download/1', methods=['POST', 'GET'])
def send_html(filename=1):
    # logger.debug("download file, path is %s" % filename)
    dirpath = basedir + "\data\images"
    filename = "789.jpg"
    return send_from_directory(dirpath, filename, as_attachment=True)


@app.route('/generate', methods=['POST'])
def getimg():
    # url=dirpath=basedir+"\data\images"+"123.png"
    # print(request.get_data())
    message = request.get_data()
    Character = {'hair color': 'random', 'hair style': 'random', 'eyes color': 'random', 'blush': 'random',
                 'smile': 'random', 'open mouth': 'random', 'hat': 'random', 'ribbon': 'random', 'glasses': 'random'}
    dirpath=os.path.join(basedir,'data','images')
    #dirpath = basedir + "\data\images"
    message = message.decode('utf-8')
    message = message.split(",")
    Character['hair color'] = message[0].split(':')[-1]
    Character['hair style'] = message[1].split(':')[-1]
    Character['eyes color'] = message[2].split(':')[-1]
    Character['blush'] = message[3].split(':')[-1]
    Character['smile'] = message[4].split(':')[-1]
    Character['open mouth'] = message[5].split(':')[-1]
    Character['hat'] = message[6].split(':')[-1]
    Character['ribbon'] = message[7].split(':')[-1]
    Character['glasses'] = message[8].split(':')[-1]
    tags=[]
    if Character.get('hair color') != 'random' and Character.get('hair color')!='0':
        tags.append(Character['hair color'])
    if Character.get('hair style') != 'random' and Character.get('hair style') != '0':
        tags.append(Character['hair style'])
    if Character.get('eyes color') != 'random' and Character.get('eyes color') != '0':
        tags.append(Character['eyes color'])
    if Character.get('blush') != 'random' and Character.get('blush')!='0':
        tags.append(Character['blush'])
    if Character.get('smile') != 'random' and Character.get('smile')!='0':
        tags.append(Character['smile'])
    if Character.get('open mouth') != 'random' and Character.get('open mouth')!='0':
        tags.append(Character['open mouth'])
    if Character.get('hat') != 'random' and Character.get('hat')!='0':
        tags.append(Character['hat'])
    if Character.get('ribbon') != 'random' and Character.get('ribbon') != '0':
        tags.append(Character['ribbon'])
    if Character.get('glasses') != 'random' and Character.get('glasses') != '0':
        tags.append(Character['glasses'])
    print(tags)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now_time = now_time.replace('-', '_')
    now_time = now_time.replace(':', '')
    now_time = now_time.replace(' ', '')
    Generate(tags,now_time)
    #time.sleep(10)
    filename = '{}.jpg'.format(now_time)
    print(filename)
    return send_from_directory(dirpath, filename, as_attachment=True)


if __name__ == '__main__':
    config = dict(
        debug=True,
        host='0.0.0.0',
        port=2000,
    )

    app.run(**config)
