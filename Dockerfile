FROM bitnami/minideb:trixie
ARG TARGETARCH

COPY target_${TARGETARCH}/ortts /usr/local/bin/ortts
RUN chmod +x /usr/local/bin/ortts

EXPOSE 12775
ENTRYPOINT ["/usr/local/bin/ortts"]
CMD ["serve", "--listen", "0.0.0.0:12775"]
STOPSIGNAL SIGTERM
