from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults

def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':
        # Receive data from client ( ajax )
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        classifier_type = str(request.POST.get('classifier_type'))

        if classifier_type == 'Naive Bayest Classifier' :
            # Unpickle model Naive Bayest
            model = pd.read_pickle(r"D:\FCI\projects\Iris-prediction\Iris-master\naive_bayest.pickle")
            # Make prediction
            result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

            classification = result[0]

            PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                    petal_width=petal_width, classification=classification)

            return JsonResponse(
                {
                    'result': classification,
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width
                },
                    safe=False)
        
        ##############################################################################
        ##############################################################################

        elif classifier_type == 'Support Vector Classifier' :
            # Unpickle model Support Vector Classifier
            model = pd.read_pickle(r"D:\FCI\projects\Iris-prediction\Iris-master\svc.pickle")
            # Make prediction
            result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

            classification = result[0]

            PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                    petal_width=petal_width, classification=classification)

            return JsonResponse(
                {
                    'result': classification,
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width
                },
                    safe=False)
        ##############################################################################
        ##############################################################################

        elif classifier_type == 'Descision Tree' :
            # Unpickle model Descision Tree
            model = pd.read_pickle(r"D:\FCI\projects\Iris-prediction\Iris-master\tree.pickle")
            # Make prediction
            result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

            classification = result[0]

            PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                    petal_width=petal_width, classification=classification)

            return JsonResponse(
                {
                    'result': classification,
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width
                },
                    safe=False)



def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
