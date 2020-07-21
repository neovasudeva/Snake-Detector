# base image
FROM mysql/mysql-server:5.7

# add sql dump file
ADD ./snake.sql /docker-entrypoint-initdb.d/snake.sql

# start mysql db
CMD ["mysqld", "--user", "mysql"]
