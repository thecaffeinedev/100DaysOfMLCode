# DL In Production
**Steps :**

`sudo docker build -t lstm:latest .`

`sudo docker images | grep lstm`

`sudo docker tag xxx dagilgon/lstm:latest`

`sudo docker login`

`sudo docker push dagilgon/lstm:latest`

`#cat del token de auth`
`sudo cat ~/.docker/config.json`


`kubectl create secret docker-registry regcred --docker-server=xxx --docker-username=login --docker-password=xxx --docker-email=login@mail.com`

`kubectl get secret regcred --output=yaml`

`kubectl get secret regcred --output="jsonpath={.data.\.dockerconfigjson}" | base64 --decode`

`pod -> pod.yaml`

`service -> service.yaml`

`deployment -> deployment.yaml`

### POD kmaster

`kubectl create -f pod.yaml`

`kubectl get pod lstm`

### POD local

`kubectl --kubeconfig ~/kubernetes/admin.conf create -f pod.yaml`

`kubectl --kubeconfig ~/kubernetes/admin.conf  get pod lstm`

#### service local

`kubectl --kubeconfig ~/kubernetes/admin.conf create -f service.yaml`

#### deployment local

`kubectl --kubeconfig ~/kubernetes/admin.conf apply -f deployment.yaml`
