CREATE TABLE `type` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `typeName` VARCHAR(200),
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
