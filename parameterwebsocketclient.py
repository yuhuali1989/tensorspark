#!/usr/local/bin/bpython
# -*- coding: utf-8 -*-
import tensorflow as tf
import mnistdnn
import higgsdnn
import moleculardnn
import math
import numpy as np
import tornado.websocket
from tornado import gen
from tornado.ioloop import IOLoop
import cStringIO

# mod (we need to find out the IP address of the param server if using the YARN-Cluster mode; see below)
# isUsingYARNclusterMode = False
isUsingYARNclusterMode = True


# TODO
# Imagenet
# Tachyon


class Borg:
  _shared_state = {}

  def __init__(self):
    self.__dict__ = self._shared_state


class TensorSparkWorker(Borg):
  def __init__(self, model_keyword, batch_size, websocket_port, yarn_ip):
    Borg.__init__(self)

    if 'model' not in self.__dict__:
      print 'Creating new Borg worker'
      if model_keyword == 'mnist':
        self.model = mnistdnn.MnistDNN(batch_size, gpu=True)
      elif model_keyword == 'higgs':
        self.model = higgsdnn.HiggsDNN(batch_size)
      elif model_keyword == 'molecular':
        self.model = moleculardnn.MolecularDNN(batch_size)
      else:
        raise
      self.batch_size = batch_size
      self.websocket_port = websocket_port
      self.loop = IOLoop.current()
      self.yarn_ip = yarn_ip
      self.loop.run_sync(self.init_websocket)
      self.iteration = 0

  @gen.coroutine
  def init_websocket(self):
    # mod
    if isUsingYARNclusterMode == False:
      self.websock = yield tornado.websocket.websocket_connect(
        "ws://localhost:%d/" % self.websocket_port, connect_timeout=3600)
      # self.websock = yield tornado.websocket.websocket_connect("ws://172.31.0.92:%d/" % self.websocket_port, connect_timeout=3600)

    # We do not know in advance Spark driver is running on which node with the YARN/Cluster mode
    else:
      import time
      import urllib2
      import json

      nodes = ({'ip-172-31-26-81.cn-north-1.compute.internal': '172.31.26.81'})
      YARN_app_queue = 'default'
      YARN_address = 'http://%s:8088' % self.yarn_ip

      YARN_app_startedTimeBegin = str(
        int(time.time() - 3600))  # We allow 3,600 sec from start of the app up to this point

      requestedURL = (
        YARN_address + '/ws/v1/cluster/apps?states=RUNNING&applicationTypes=SPARK&limit=1' + '&queue=' + YARN_app_queue + '&startedTimeBegin=' + YARN_app_startedTimeBegin)
      print 'Sent request to YARN: ' + requestedURL
      response = urllib2.urlopen(requestedURL)
      html = response.read()
      responseJson = json.loads(html)
      amHostPortFromJson = responseJson['apps']['app'][0]['amHostHttpAddress']
      amHostFromJson = str(amHostPortFromJson).split(':')[0]
      print 'HTML 111111111:' + html
      # amHost_start = html.find('amHostHttpAddress') + len('amHostHttpAddress":"')
      # print 'amHost_start 111111111:%d' % amHost_start
      # amHost_length = len('ip-172-31-19-76.cn-north-1.compute.internal')
      # amHost = html[amHost_start: amHost_start + amHost_length]
      print 'amHostHttpAddress is: ' + amHostFromJson
      # print 'amHostHttpIpis: ' + nodes[amHost]
      try:
        self.websock = yield tornado.websocket.websocket_connect(
          "ws://%s:%d/" % (amHostFromJson, self.websocket_port), connect_timeout=3600)
        print 'Connected to server running on %s' % amHostFromJson
      except:
        print 'Could not connect to server on %s' % amHostFromJson
        # mod: end

  def train_partition(self, partition):
    while True:
      # print 'TensorSparkWorker().train_partition iteration %d' % self.iteration

      labels, features = self.model.process_partition(partition)

      if len(labels) is 0:
        break

      if self.time_to_pull(self.iteration):
        self.request_parameters()

      self.model.train(labels, features)
      self.iteration += 1

      if self.time_to_push(self.iteration):
        self.push_gradients()

    return []
    # return [self.train(x) for x in partition]

  def test_partition(self, partition):
    labels, features = self.model.process_partition(partition)
    self.request_parameters()
    error_rate = self.model.test(labels, features)
    return [error_rate]
    # return [self.test(x) for x in partition]

  def test(self, data):
    # print 'TensorSparkWorker().test "%s"' % data
    if len(data) is 0:
      return 1.0
    self.request_parameters()
    accuracy = self.model.test(data)
    return accuracy

  #      self.model.

  def time_to_pull(self, iteration):
    return iteration % 5 == 0

  #      return True

  def time_to_push(self, iteration):
    return iteration % 5 == 0

  #      return True

  def request_parameters(self):
    IOLoop.current().run_sync(self.request_parameters_coroutine)

  @gen.coroutine
  def request_parameters_coroutine(self):
    # more like receive parameters now
    parameters = yield self.websock.read_message()
    parameters = self.model.deserialize(parameters)
    self.model.assign_parameters(parameters)

  def push_gradients(self):
    IOLoop.current().run_sync(self.push_gradients_coroutine)

  @gen.coroutine
  def push_gradients_coroutine(self):
    gradients = self.model.get_gradients()
    serialized = self.model.serialize(self.model.get_gradients())
    del gradients
    self.websock.write_message(serialized, binary=True)
