from flask import Flask, render_template, request, redirect, url_for, Response, flash, send_from_directory
import json
import importlib
import sys
import os 
import glob
import time
from LibSearch import *
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = Flask(__name__,
            static_url_path='', 
            static_folder='front/static',
            template_folder='front/templates')
app.secret_key = b'p3/'

@app.route('/')
def home():
   return render_template('buscador.html')


@app.route('/search', methods = ['POST'])
def search():
   type_of_search = request.form.get("type")

   if type_of_search == "knn":
      k = int(request.form.get("k"))
   elif type_of_search == "range":
      radius = float(request.form.get("radius"))
   else:
      return redirect(url_for('home'))


   if 'file' not in request.files:
      flash('No hay archivo', 'alert-danger')
      return redirect(url_for('home'))
   
   file = request.files['file']

   if file.filename == '':
      flash('Archivo no seleccionado', 'alert-danger')
      return redirect(url_for('home'))
   
   if file:
      filename = secure_filename(file.filename)

   files = glob.glob('input/*')
   for f in files:
      os.remove(f)
   
   start = time.time()
   archivo =  open("input/"+str(file.filename), "wb")
   archivo.write(file.read())

   if type_of_search == "range":
      list_of_path = range_search_rtree(str(file.filename), radius, True)
      end = time.time()
      flash(u'Tiempo: ' + str(end - start) + ' segundos',  'alert-success')
   
   else:
      list_of_path = KNN_FaceRecognition(str(file.filename), k, True)
      end = time.time()
      flash(u'Tiempo: ' + str(end - start) + ' segundos',  'alert-success')
   


   images_output = list()
   for file in list_of_path:
      images_output.append('source/' + file)
      
   print(images_output)
   return render_template('buscador.html', images_output=images_output)



@app.route('/source/<path:filename>')
def base_static(filename):
   return send_from_directory('source/', filename)
   
if __name__ == '__main__':
   app.run()
