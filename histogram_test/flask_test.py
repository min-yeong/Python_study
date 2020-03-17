from flask import Flask, redirect, url_for, request, render_template, jsonify
import os
import cv2
import histogram
app = Flask(__name__)

@app.route("/")
def jsonlisttest():
    searchimg = cv2.imread('D:/python/histogram_test/searchimage/wahoo_test.jpg', cv2.IMREAD_COLOR)
    findlist_x = histogram.histogram(searchimg)
    lst = histogram.matching(searchimg, findlist_x)
    
    return jsonify(lst)

if __name__ == '__main__':
    app.run(debug=True)