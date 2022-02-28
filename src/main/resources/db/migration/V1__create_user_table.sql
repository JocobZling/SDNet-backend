CREATE TABLE `user` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `name` VARCHAR(200),
    `email`VARCHAR(200),
    `password` VARCHAR(200),
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
