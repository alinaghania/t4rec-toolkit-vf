T4Rec import OK
2025-09-01 11:57:01 | == STEP 0: CHECK df_events en mémoire ==
2025-09-01 11:57:01 | df_events shape: (198873, 3)
2025-09-01 11:57:01 | df_events dtypes:
NUMTECPRS                         float64
DATMAJ                     datetime64[ns]
SOUSCRIPTION_PRODUIT_1M            object
dtype: object
2025-09-01 11:57:01 | Aperçu df_events.head():
   NUMTECPRS     DATMAJ SOUSCRIPTION_PRODUIT_1M
0       83.0 2024-01-31      Aucune_Proposition
1      360.0 2024-01-31      Aucune_Proposition
2      400.0 2024-01-31      Aucune_Proposition
3      534.0 2024-01-31      Aucune_Proposition
4      556.0 2024-01-31      Aucune_Proposition
2025-09-01 11:57:01 | Stats rapides — rows=198,873 | clients uniques=198,873 | mois distincts=1
2025-09-01 11:57:01 | TOP valeurs de la cible (head 10):
Aucune_Proposition                 192793
Livret_Epargne                       1295
Credit_Consommation_Hors_Budget       946
Carte_Paiement_Cleo                   575
Assurance_Vie                         495
Carte_Paiement_Premier                406
Assurance_Prevoyance                  355
Credit_Decouvert_Autorise             352
Assurance_Habitation                  297
Credit_Personnel_Budget               269
Name: SOUSCRIPTION_PRODUIT_1M, dtype: int64
2025-09-01 11:57:01 | Présence des valeurs à exclure dans df_events: []
2025-09-01 11:57:01 | STEP 0 DONE in 0.1s
2025-09-01 11:57:01 | == STEP A: FS (profil) ==
2025-09-01 11:57:01 | FS: .fit() sur DATASET_MAIN...
ERROR:dataiku.core.dataset_write:Exception caught while writing
Traceback (most recent call last):
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 353, in run
    self.streaming_api.wait_write_session(self.session_id)
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 296, in wait_write_session
    raise Exception(u'An error occurred during dataset write (%s): %s' % (id, decoded_resp["message"]))
Exception: An error occurred during dataset write (96VwszCIju): DSQuotaExceededException: The DiskSpace quota of /data/lab/dcc is exceeded: quota = 131941395333120 B = 120 TB but diskspace consumed = 131941484158365 B = 120.00 TB
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyStoragespaceQuota(DirectoryWithQuotaFeature.java:200)
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyQuota(DirectoryWithQuotaFeature.java:227)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.verifyQuota(FSDirectory.java:1242)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1074)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1033)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.addBlock(FSDirWriteFileOp.java:512)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.saveAllocatedBlock(FSDirWriteFileOp.java:779)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.storeAllocatedBlock(FSDirWriteFileOp.java:259)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.getAdditionalBlock(FSNamesystem.java:2840)
	at org.apache.hadoop.hdfs.server.namenode.NameNodeRpcServer.addBlock(NameNodeRpcServer.java:874)
	at org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolServerSideTranslatorPB.addBlock(ClientNamenodeProtocolServerSideTranslatorPB.java:589)
	at org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos$ClientNamenodeProtocol$2.callBlockingMethod(ClientNamenodeProtocolProtos.java)
	at org.apache.hadoop.ipc.ProtobufRpcEngine$Server$ProtoBufRpcInvoker.call(ProtobufRpcEngine.java:533)
	at org.apache.hadoop.ipc.RPC$Server.call(RPC.java:1070)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:989)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:917)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAs(Subject.java:422)
	at org.apache.hadoop.security.UserGroupInformation.doAs(UserGroupInformation.java:1898)
	at org.apache.hadoop.ipc.Server$Handler.run(Server.java:2894)

2025-09-01 11:59:26 | FS OK. seq_cols_fs=['AGE', 'NBRMMSANCCLI', 'NB_Flux_Deb_12DM', 'NB_MOIS_DEPUIS_DERNIER_EP_LOGEMENT', 'MNTECSCPTDPT', 'MNTECRCRD', 'NB_RESTAURATION_12DM', 'NB_Ope_Carte_12DM', 'NBCHQEMIGLISS_M12', 'VAR_MNT_EPARGNE_9M'] | cat_cols_fs=['LIBFAMCSP', 'MEILLEURE_CARTE_DETENUE_M', 'CONNAISSANCE_MIF', 'FILTER_TRAIN_TEST', 'DATMAJ']
2025-09-01 11:59:26 | Profil retenu — cat: ['MEILLEURE_CARTE_DETENUE_M', 'LIBFAMCSP', 'CONNAISSANCE_MIF']
2025-09-01 11:59:26 | Profil retenu — num: ['AGE']
2025-09-01 11:59:26 | STEP A DONE in 145.2s
2025-09-01 11:59:26 | == STEP B: SAVE df_events -> Dataiku dataset ==
2025-09-01 11:59:33 | Écriture df_event échouée: An error occurred during dataset write (96VwszCIju): DSQuotaExceededException: The DiskSpace quota of /data/lab/dcc is exceeded: quota = 131941395333120 B = 120 TB but diskspace consumed = 131941484158365 B = 120.00 TB
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyStoragespaceQuota(DirectoryWithQuotaFeature.java:200)
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyQuota(DirectoryWithQuotaFeature.java:227)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.verifyQuota(FSDirectory.java:1242)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1074)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1033)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.addBlock(FSDirWriteFileOp.java:512)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.saveAllocatedBlock(FSDirWriteFileOp.java:779)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.storeAllocatedBlock(FSDirWriteFileOp.java:259)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.getAdditionalBlock(FSNamesystem.java:2840)
	at org.apache.hadoop.hdfs.server.namenode.NameNodeRpcServer.addBlock(NameNodeRpcServer.java:874)
	at org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolServerSideTranslatorPB.addBlock(ClientNamenodeProtocolServerSideTranslatorPB.java:589)
	at org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos$ClientNamenodeProtocol$2.callBlockingMethod(ClientNamenodeProtocolProtos.java)
	at org.apache.hadoop.ipc.ProtobufRpcEngine$Server$ProtoBufRpcInvoker.call(ProtobufRpcEngine.java:533)
	at org.apache.hadoop.ipc.RPC$Server.call(RPC.java:1070)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:989)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:917)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAs(Subject.java:422)
	at org.apache.hadoop.security.UserGroupInformation.doAs(UserGroupInformation.java:1898)
	at org.apache.hadoop.ipc.Server$Handler.run(Server.java:2894)

Traceback (most recent call last):
  File "<ipython-input-4-90b51a802747>", line 153, in <cell line: 150>
    out_ds.write_with_schema(df_events)
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset.py", line 1133, in write_with_schema
    self.write_dataframe(df, True, drop_and_create)
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset.py", line 1175, in write_dataframe
    writer.write_dataframe(df)
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 600, in __exit__
    self.close()
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 594, in close
    self.waiter.wait_end()
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 344, in wait_end
    self.raise_on_failure()
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 329, in raise_on_failure
    raise self.exception
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 353, in run
    self.streaming_api.wait_write_session(self.session_id)
  File "/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py", line 296, in wait_write_session
    raise Exception(u'An error occurred during dataset write (%s): %s' % (id, decoded_resp["message"]))
Exception: An error occurred during dataset write (96VwszCIju): DSQuotaExceededException: The DiskSpace quota of /data/lab/dcc is exceeded: quota = 131941395333120 B = 120 TB but diskspace consumed = 131941484158365 B = 120.00 TB
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyStoragespaceQuota(DirectoryWithQuotaFeature.java:200)
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyQuota(DirectoryWithQuotaFeature.java:227)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.verifyQuota(FSDirectory.java:1242)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1074)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1033)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.addBlock(FSDirWriteFileOp.java:512)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.saveAllocatedBlock(FSDirWriteFileOp.java:779)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.storeAllocatedBlock(FSDirWriteFileOp.java:259)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.getAdditionalBlock(FSNamesystem.java:2840)
	at org.apache.hadoop.hdfs.server.namenode.NameNodeRpcServer.addBlock(NameNodeRpcServer.java:874)
	at org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolServerSideTranslatorPB.addBlock(ClientNamenodeProtocolServerSideTranslatorPB.java:589)
	at org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos$ClientNamenodeProtocol$2.callBlockingMethod(ClientNamenodeProtocolProtos.java)
	at org.apache.hadoop.ipc.ProtobufRpcEngine$Server$ProtoBufRpcInvoker.call(ProtobufRpcEngine.java:533)
	at org.apache.hadoop.ipc.RPC$Server.call(RPC.java:1070)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:989)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:917)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAs(Subject.java:422)
	at org.apache.hadoop.security.UserGroupInformation.doAs(UserGroupInformation.java:1898)
	at org.apache.hadoop.ipc.Server$Handler.run(Server.java:2894)


---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
<ipython-input-4-90b51a802747> in <cell line: 150>()
    151     # Si le dataset n'existe pas dans le Flow, crée-le d'abord dans l'UI (tu l'as déjà fait).
    152     out_ds = dataiku.Dataset(EVENTS_TMP)
--> 153     out_ds.write_with_schema(df_events)
    154     log(f"Saved events -> {EVENTS_TMP} : {df_events.shape}")
    155 except Exception as e:

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset.py in write_with_schema(self, df, drop_and_create, **kwargs)
   1131                              "dataframe as argument. You gave a %s" %
   1132                              (df is None and "None" or df.__class__))
-> 1133         self.write_dataframe(df, True, drop_and_create)
   1134 
   1135     def write_dataframe(self, df, infer_schema=False, drop_and_create=False, **kwargs):

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset.py in write_dataframe(self, df, infer_schema, drop_and_create, **kwargs)
   1173 
   1174             with self.get_writer() as writer:
-> 1175                 writer.write_dataframe(df)
   1176 
   1177         except AttributeError as e:

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in __exit__(self, type, value, traceback)
    598 
    599     def __exit__(self, type, value, traceback):
--> 600         self.close()

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in close(self)
    592         self.remote_writer.flush()
    593         self.remote_writer.close()
--> 594         self.waiter.wait_end()
    595 
    596     def __enter__(self,):

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in wait_end(self)
    342         """
    343         self.join()
--> 344         self.raise_on_failure()
    345 
    346     def run(self):

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in raise_on_failure(self)
    327         if self.exception_type is not None:
    328             if (sys.version_info > (3, 0)):
--> 329                 raise self.exception
    330             else:
    331                 exec("raise self.exception_type, self.exception, self.traceback")

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in run(self)
    351             if self.session_id == MISSING_ID_MARKER and self.session_init_message is not None:
    352                 raise Exception(u'An error occurred while starting the writing to the dataset : %s' % self.session_init_message)
--> 353             self.streaming_api.wait_write_session(self.session_id)
    354         except Exception as e:
    355             logger.exception("Exception caught while writing")

/home/dataiku/dataiku-dss-12.6.4/python/dataiku/core/dataset_write.py in wait_write_session(self, id)
    294             print ("%s rows successfully written (%s)" % (writtenRows,id))
    295         else:
--> 296             raise Exception(u'An error occurred during dataset write (%s): %s' % (id, decoded_resp["message"]))
    297 
    298     def push_data(self,id,generator):

Exception: An error occurred during dataset write (96VwszCIju): DSQuotaExceededException: The DiskSpace quota of /data/lab/dcc is exceeded: quota = 131941395333120 B = 120 TB but diskspace consumed = 131941484158365 B = 120.00 TB
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyStoragespaceQuota(DirectoryWithQuotaFeature.java:200)
	at org.apache.hadoop.hdfs.server.namenode.DirectoryWithQuotaFeature.verifyQuota(DirectoryWithQuotaFeature.java:227)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.verifyQuota(FSDirectory.java:1242)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1074)
	at org.apache.hadoop.hdfs.server.namenode.FSDirectory.updateCount(FSDirectory.java:1033)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.addBlock(FSDirWriteFileOp.java:512)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.saveAllocatedBlock(FSDirWriteFileOp.java:779)
	at org.apache.hadoop.hdfs.server.namenode.FSDirWriteFileOp.storeAllocatedBlock(FSDirWriteFileOp.java:259)
	at org.apache.hadoop.hdfs.server.namenode.FSNamesystem.getAdditionalBlock(FSNamesystem.java:2840)
	at org.apache.hadoop.hdfs.server.namenode.NameNodeRpcServer.addBlock(NameNodeRpcServer.java:874)
	at org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolServerSideTranslatorPB.addBlock(ClientNamenodeProtocolServerSideTranslatorPB.java:589)
	at org.apache.hadoop.hdfs.protocol.proto.ClientNamenodeProtocolProtos$ClientNamenodeProtocol$2.callBlockingMethod(ClientNamenodeProtocolProtos.java)
	at org.apache.hadoop.ipc.ProtobufRpcEngine$Server$ProtoBufRpcInvoker.call(ProtobufRpcEngine.java:533)
	at org.apache.hadoop.ipc.RPC$Server.call(RPC.java:1070)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:989)
	at org.apache.hadoop.ipc.Server$RpcCall.run(Server.java:917)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAs(Subject.java:422)
	at org.apache.hadoop.security.UserGroupInformation.doAs(UserGroupInformation.java:1898)
	at org.apache.hadoop.ipc.Server$Handler.run(Server.java:2894)


