FROM ubuntu:26.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (LaTeX for diagramming, poppler for images, curl for babashka)
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    python3 \
    python3-pip \
    python3-venv \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-extra-utils \
    texlive-fonts-recommended \
    poppler-utils \
    curl \
    default-jre \
    iverilog \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install official Clojure CLI tools (True JVM evaluation)
RUN curl -O https://download.clojure.org/install/linux-install-1.11.1.1413.sh && \
    chmod +x linux-install-1.11.1.1413.sh && \
    ./linux-install-1.11.1.1413.sh && \
    rm linux-install-1.11.1.1413.sh

# Install Babashka (Fast native Clojure runtime, just in case)
RUN curl -sLO https://raw.githubusercontent.com/babashka/babashka/master/install && \
    chmod +x install && \
    ./install --dir /usr/local/bin && \
    rm install

# Create user 'b' with password 'b' and add to sudoers
RUN useradd -m -s /bin/bash b && \
    echo "b:b" | chpasswd && \
    usermod -aG sudo b

USER b
WORKDIR /home/b

# Clone the repository
RUN git clone https://github.com/mingusb/transformer-golf.git

WORKDIR /home/b/transformer-golf

# Create a virtual environment and install python dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir torch egglog numpy scikit-learn triton Mathics3 Mathics3_Scanner

# Set the default command to run the entire build pipeline
CMD ["/bin/bash", "-c", "source venv/bin/activate && ./build.sh"]
