CREATE TABLE `photoType` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `mainType` VARCHAR(200),
    `typeList` text,
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
