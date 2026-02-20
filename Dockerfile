FROM 958146224438.dkr.ecr.us-west-2.amazonaws.com/docker-library/python3.12

RUN apt-get update \
    && apt-get install -y libc6 \
                          gcc \
                          g++ \
                          cargo \
                          cpp \
                          cmake \
                          pkg-config \
                          libpng-dev \
                          libfreetype6 \
                          libfreetype6-dev \
                          fonts-liberation \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm ~/.cache/matplotlib -rf

RUN mkdir -p /opt/datasci-season-renewal
COPY requirements.txt /opt/
RUN uv pip install -r /opt/requirements.txt --no-cache-dir

# copy code in image
COPY . /opt/data-science

WORKDIR /opt/data-science/slt_season_renewal

ENTRYPOINT ["python", "/opt/data-science/slt_season_renewal/slt_season_renewal.py"]