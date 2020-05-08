FROM ubuntu:16.04
RUN apt-get update -y && \
  apt-get install -y python-pip python-dev unzip && \
  pip install --upgrade pip

WORKDIR /app

# We copy just the requirements.txt first to leverage Docker cache
COPY ./neuraltalk2/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./neuraltalk2 .

# RUN wget -O pre_trained_model.zip https://uc54170c363eda96b1dbb77029d4.dl.dropboxusercontent.com/cd/0/get/Ay9lhpurh6hd30GSNHC-ZiX4LD64BBfTElx2Cr1npcttX4t1F4azLvATQqFcc-Dh9O6iE1Nlk0U786JXx6YCmK6GyBb9yk_rdRMXKtxMEBWEA51kG0lEcdS0ThOrt0RnGhU/file?_download_id=85388372573420495429059149766284181497394146010749638288403808974&_notify_domain=www.dropbox.com&dl=1 && \
#   wget -O vocab.zip https://ucc76873a825745e4af95ec96cb9.dl.dropboxusercontent.com/cd/0/get/Ay9OaXpTlfW2vy2CAYq2QJIMThAKO-TqMIPs2cJVAJEc-izt_OPVoiqOOcSfvvMBbfDq-utYL-jdO6AUiAtdzyQRRgHZVQm2SplxufkO01Cr2VqE368_yMRjXZlNZxX9TYc/file?_download_id=3760584107917515484540467727693935852244846986662611208432831207&_notify_domain=www.dropbox.com&dl=1

COPY resources/vocap.zip vocab.zip;
COPY resource/pretrained_mode.zip .
RUN unzip vocab.zip -d data && unzip pre_trained_model.zip -d models/

ENTRYPOINT [ "python" ]

CMD [ "sample.py", "--image='png/example.png'" ]