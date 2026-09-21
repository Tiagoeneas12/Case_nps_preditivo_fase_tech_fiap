# Dicionário de Dados — Case NPS Preditivo

Base de dados com histórico de pedidos, entregas e interações de atendimento ao cliente, utilizada no Tech Challenge Fase 1.

| Campo | Descrição | Tipo sugerido | Domínio / Observações |
|---|---|---|---|
| `customer_id` | Identificador único do cliente | string/ID | Chave de identificação do cliente |
| `order_id` | Identificador único do pedido | string/ID | Chave de identificação do pedido |
| `customer_age` | Idade do cliente | inteiro | Anos |
| `customer_region` | Região geográfica do cliente | categórico | Ex.: Norte, Nordeste, Sul, Sudeste, Centro-Oeste |
| `customer_tenure_months` | Tempo de relacionamento do cliente com a empresa | inteiro | Em meses |
| `order_value` | Valor total do pedido | numérico (float) | Moeda (R$) |
| `items_quantity` | Quantidade de itens no pedido | inteiro | Unidades |
| `discount_value` | Valor de desconto aplicado ao pedido | numérico (float) | Moeda (R$) |
| `payment_installments` | Número de parcelas do pagamento | inteiro | Quantidade de parcelas |
| `delivery_time_days` | Tempo total de entrega | inteiro | Dias |
| `delivery_delay_days` | Quantidade de dias de atraso na entrega | inteiro | Dias (pode ser 0 ou negativo se entregue antes do prazo, a depender da base) |
| `freight_value` | Valor do frete | numérico (float) | Moeda (R$) |
| `delivery_attempts` | Número de tentativas de entrega | inteiro | Quantidade de tentativas |
| `customer_service_contacts` | Número de contatos do cliente com o atendimento | inteiro | Quantidade de contatos |
| `resolution_time_days` | Tempo para resolução de problemas | inteiro | Dias |
| `complaints_count` | Número de reclamações registradas pelo cliente | inteiro | Quantidade de reclamações |
| `repeat_purchase_30d` | Indica se houve recompra em até 30 dias após o pedido | binário | 0 = não, 1 = sim |
| `csat_internal_score` | Score interno de satisfação do cliente | numérico | Escala interna da empresa (verificar range na base) |
| `nps_score` | Nota de satisfação do cliente (NPS), coletada após a experiência de compra | inteiro | Escala de 0 a 10 — **variável alvo (target)** |

## Observações gerais

- A variável `nps_score` é a variável alvo (target) do problema de negócio, representando a satisfação do cliente em uma escala de 0 a 10.
- Classificação tradicional de NPS a partir da nota:
  - **Detratores**: notas de 0 a 6
  - **Neutros**: notas 7 e 8
  - **Promotores**: notas 9 e 10
- Campos relacionados a **logística** (`delivery_time_days`, `delivery_delay_days`, `freight_value`, `delivery_attempts`) e **atendimento** (`customer_service_contacts`, `resolution_time_days`, `complaints_count`) são candidatos fortes a variáveis explicativas (features) para análise exploratória e modelagem preditiva.
- `csat_internal_score` pode servir como variável de apoio/cruzamento com o NPS, mas não deve ser confundido com a variável alvo.
