cd studio

npm install

npm run build

export HAIKU_STUDIO_PORT=39177
export HAIKU_STUDIO_EXTERNAL_BACKEND=1
export HAIKU_STUDIO_ALLOW_SHUTDOWN=1
export NODE_ENV=production

npm run server &
npm run studio
