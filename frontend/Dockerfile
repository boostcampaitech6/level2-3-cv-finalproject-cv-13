FROM node:19.6.1-alpine3.17 as builder

ARG REACT_APP_DPR_ENDPOINT

WORKDIR /app
COPY . /app


RUN echo "REACT_APP_DPR_ENDPOINT=${REACT_APP_DPR_ENDPOINT}" >> .env

RUN npm install react@17.0.0 react-dom@17.0.0
RUN npm install --force
RUN npm run build

RUN rm .env

FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/conf.d

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]