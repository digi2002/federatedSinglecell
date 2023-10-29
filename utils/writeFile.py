#save

from collections import Counter



def outResults(resultdir, dataset,modelname,comms_round,num_clients,results,app,randomint,times =[]):
    outfile = resultdir + dataset + '_' + modelname + '_commround_' + str(comms_round) + '_clientNum_' + str(num_clients) + '_'+app+'_'+str(randomint)+'.txt'
    print('length of times'+ str(len(times)))
    if len(times) != 0:
        outfile = outfile + '_time'
    print(outfile)
    #outfile = resultdir + dataset + '_' + modelname + '_commround_' + str(comms_round) + '_clientNum_' + str(num_clients) + '_'+app+'_'+str(randomint)+'.txt'
    output = open(outfile,'w')
    output.write('center\n')
    output.write(str(results['center_acc'])+','+str(results['center_f1'])+'\n')
    output.write('local\n')
    output.write(str(results['local_acc']) + ',' + str(results['local_f1'])+'\n')
    output.write('global\n')
    output.write(str(results['global_acc']) + ',' + str(results['global_f1'])+'\n')
    if len(times)!= 0:
        center_time, local_time, global_time = times
        output.write('times\n')
        output.write(str(center_time) + ',' + str(local_time) + ',' + str(global_time) + '\n')




def outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients,randomint):
    outfile = resultdir + dataset + '_' + modelname + '_commround_' + str(comms_round) + '_clientNum_' + str(
        num_clients) + '_'+str(randomint)+'_sta.txt'
    output = open(outfile, 'w')
    output.write('y\n')
    output.write(str(Counter(y)) + '\n')

    output.write('y_train\n')
    output.write(str(Counter(y_train))+'\n')
    output.write('y_test\n')
    output.write(str(Counter(y_test)) + '\n')
    for client_name in clients.keys():
        client_data = clients[client_name]
        X_train, y_train = list(map(list, zip(*client_data)))
        output.write(client_name+'\n')
        output.write(str(Counter(y_train))+'\n')

