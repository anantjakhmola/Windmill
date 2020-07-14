from watson_machine_learning_client import WatsonMachineLearningAPIClient
import math
import PIL
from PIL import Image
import numpy as np
from flask import Flask, request, json, jsonify
import os

#
# 1.  Fill in wml_credentials.
#
wml_credentials = {
  "instance_id": "",
  "password": "",
  "url": "",
  "username": ""
}

client = WatsonMachineLearningAPIClient( wml_credentials )

#
# 2.  Fill in one or both of these:
#     - model_deployment_endpoint_url
#     - function_deployment_endpoint_url
#
model_deployment_endpoint_url    = "";
function_deployment_endpoint_url = "";

def createPayload( canvas_data ):
    dimension      = canvas_data["height"]
    img            = Image.fromarray( np.asarray( canvas_data["data"] ).astype('uint8').reshape( dimension, dimension, 4 ), 'RGBA' )
    sm_img         = img.resize( ( 28, 28 ), Image.LANCZOS )
    alpha_arr      = np.array( sm_img.split()[-1] )
    norm_alpha_arr = alpha_arr / 255
    payload_arr    = norm_alpha_arr.reshape( 1, 784 )
    payload_list   = payload_arr.tolist()
    return { "values" : payload_list }


app = Flask( __name__, static_url_path='' )

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int( os.getenv( 'PORT', 8000 ) )

@app.route('/')
def root():
    return app.send_static_file( 'index.html' )

@app.route( '/sendtomodel', methods=['POST'] )
def sendtomodel():
    try:
        print( "sendtomodel..." )
        if model_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = canvas_data
            result = client.deployments.score( model_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Model endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )
    
@app.route( '/sendtofunction', methods=['POST'] )
def sendtofunction():
    try:
        print( "sendtofunction..." )
        if function_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = canvas_data
            result = client.deployments.score( function_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Function endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )

@app.route( '/sendtowebserver', methods=['POST'] )
def sendtowebserver():
    try:
        print( "sendtowebserver..." )
        if model_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = createPayload( canvas_data )
            result = client.deployments.score( model_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Model endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=port, debug=True)
